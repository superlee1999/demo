#include "Tracking/Tracker.h"
#include "Tracking/LinearAssignment.h"
#include <algorithm>
#include <numeric>
#include <iterator>
#include <functional>
#include <boost/lexical_cast.hpp>
#include <boost/foreach.hpp>
#include <limits>
//#include "XML/XMLReader.h"
//#include "XML/XMLBase.h"
#include "Utils/Exception.h"
#include "Utils/CVROI.h"
#include <boost/bind.hpp>
#include "Serialisation/json.h"

using namespace std;
using namespace cv;
using namespace Tracking::LA;
//using namespace XML;
using namespace Utils;

namespace {
	template<typename T>
    Point_<T> BoxCentre (const cv::Rect_<T>& box) 
    {
        return Point_<T>((box.br().x+box.tl().x)/2, (box.br().y+box.tl().y)/2);
    }

	template<typename T>
	bool IsBoxesClosed(cv::Rect_<T> box1, cv::Rect_<T> box2, const cv::Size& extension = cv::Size(5, 5))
	{
		// find the bigger box, and always set box1 as the bigger one
		if (box1.width*box1.height < box2.width*box2.height)
			std::swap(box1, box2);

		int boxWidth = box2.width/2;
		int boxHeight = box2.height/2;
		boxWidth += extension.width;
		boxHeight += extension.height;
		box1.width += 2*boxWidth;
		box1.height += 2*boxHeight;
		box1.x -= boxWidth;
		box1.y -= boxHeight;

		return box1.contains(cv::Point(BoxCentre(box2)));
	}

	Rect MergeBoxesImpl(const cv::Rect& box1, const cv::Rect& box2)
	{
		int tlx = std::min(box1.tl().x, box2.tl().x);
		int tly = std::min(box1.tl().y, box2.tl().y);
		int brx = std::max(box1.br().x, box2.br().x);
		int bry = std::max(box1.br().y, box2.br().y);

		return Rect(tlx, tly, brx - tlx + 1, bry - tly + 1);
	}

	std::vector<cv::Rect> MergeBoxes(const std::vector<cv::Rect>& boxes, const cv::Size& extension = cv::Size(5, 5))
	{
		int numBoxes = boxes.size();
		if (numBoxes == 0)
			return vector<Rect>();

		std::unique_ptr<int[]> mergedIndex(new int[numBoxes]());
		vector<Rect> mergedBoxes(boxes.begin(), boxes.end());

		for (int i = 0; i < numBoxes - 1; ++i)
		{
			for (int j = i + 1; j < numBoxes; ++j)
			{
				if (mergedIndex[i] == 0 && mergedIndex[j] == 0 && IsBoxesClosed(boxes[i], boxes[j], extension))
				{
					mergedBoxes[i] = MergeBoxesImpl(mergedBoxes[i], mergedBoxes[j]);
					mergedIndex[j] = 1;
				}
			}
		}

		// remove redundancy
		int index = 0;
		mergedBoxes.erase(std::remove_if(mergedBoxes.begin(), mergedBoxes.end(), [&mergedIndex, index](const cv::Rect& box) mutable { return mergedIndex[index++]; }), mergedBoxes.end());

		return mergedBoxes;
	}

	enum class EAxis { X, Y };
	bool CoordinateInFrame(const cv::Size& frameSize, int coordinate, EAxis axis)
	{
		switch (axis)
		{
		case EAxis::X:
			return coordinate >= 0 && coordinate < frameSize.width ? true : false;
		case EAxis::Y:
			return coordinate >= 0 && coordinate < frameSize.height ? true : false;
		default:
			return false;
		}
	}

	int CalcHistogramIndex(const cv::Vec3b& colour, int numBins)
	{
		int index = 0;
		int I[3] = {0};
		int binValue = 256 / numBins;
		for (int k = 0; k < 3; ++k)
		{
			I[k] = static_cast<int>(std::floor(colour.val[k] / binValue));
			if (k < 2)
				index += I[k] * static_cast<int>(std::pow(numBins, 2 - k));
			else
				index += I[k];
		}

		return index;
	}


	struct SCostCalc
    {
        SCostCalc(const cv::Rect_<float>& box, float alpha)
			: m_centre(BoxCentre(box))
			, m_height(box.height)
			, m_width(box.width)
			, m_alpha(alpha) 
		{
		}

        template <typename T>
        float operator()(const cv::Rect_<T>& t) const
        {
			auto tCentre = BoxCentre(t);
			return m_alpha * std::pow((m_centre.x-tCentre.x), 2) + std::pow((m_centre.y-tCentre.y), 2) + 
				(1-m_alpha) * std::abs((m_width-t.width)*(m_height-t.height));
        }

    private:
		cv::Point_<float> m_centre;
		float m_height;
		float m_width;
        float m_alpha;
    };
	

}


namespace Tracking {
    namespace Tracker {
		
		bool IsBoxCloseToEntries(const cv::Rect& box, const cv::Size frameSize, const std::vector<cv::Rect>* definedEntries)
		{
			// check if quater box is close to the frame boundaries
			Rect qbox(box.x + box.width / 4, box.y + box.height, box.width / 2, box.height / 2);
			if (!CoordinateInFrame(frameSize, qbox.tl().x, EAxis::X) || !CoordinateInFrame(frameSize, qbox.br().x, EAxis::X) ||
				!CoordinateInFrame(frameSize, qbox.tl().y, EAxis::Y) || !CoordinateInFrame(frameSize, qbox.br().y, EAxis::X))
				return true;

			// actually check if box centre close to entries
			cv::Point centre = BoxCentre(box);
			// check if close to any predefined exit points
			if (definedEntries != nullptr && !definedEntries->empty())
			{
				for (const auto& entry: *definedEntries)
				{
					if (entry.contains(centre))
						return true;
				}
			}

			return false;
		}

		bool IsBoxesCloseToEntries(const std::vector<cv::Rect>& boxes, const cv::Size frameSize, const std::vector<cv::Rect>* definedEntries)
		{
			bool retval = true;
			for (auto& box: boxes)
			{
				retval = retval && IsBoxCloseToEntries(box, frameSize, definedEntries);
			}

			return retval;
		}
		/****************************************/
		// Kalman tracker
		/****************************************/
		SKalmanParam::SKalmanParam(const std::string& configFile)
		{
			CREQUIRE(!configFile.empty(), "No configuration file specified");

			Json::Reader reader;
			Json::Value root;
			std::ifstream ifs(configFile.c_str());
			CREQUIRE(ifs.is_open(), "Error opening configuration file: " + configFile);
			CREQUIRE(reader.parse(ifs, root), "Error parsing configuration file: " + reader.getFormattedErrorMessages());

			const auto& config = root["Tracking"]["KalmanFilter"];
			
			m_maxOverlapPropagate = config["MaxOverlapPropagation"].asInt();
			m_maxLostPropagate = config["MaxLostPropagation"].asInt();
			m_maxPropagateNoRecord = config["MaxPropagationNoRecord"].asInt();
		}

		std::unique_ptr<const SKalmanParam> CKalmanFilter::m_kalmanParam(nullptr);
		void CKalmanFilter::SetGlobalConfig(const std::string& configFile)
		{
			m_kalmanParam.reset(new SKalmanParam(configFile));
		}

		void CKalmanFilter::Init(const cv::Rect& box)
		{
			if (!m_kalmanParam)
				throw CException("Global configuration parameters are not set");

			m_tracker.init(STATE_D, MEASURE_D);
			// set transition matrix
			setIdentity(m_tracker.transitionMatrix, Scalar_<float>(1.0f));
			m_tracker.transitionMatrix.at<float>(0,4) = 1.0f;
			m_tracker.transitionMatrix.at<float>(1,5) = 1.0f;
			m_tracker.transitionMatrix.at<float>(2,6) = 1.0f;
			m_tracker.transitionMatrix.at<float>(3,7) = 1.0f;
            // set measurement matrix
			setIdentity(m_tracker.measurementMatrix, Scalar_<float>(1.0f));
			// set noise cov
            setIdentity(m_tracker.processNoiseCov, Scalar_<float>::all(1e-4f));
            setIdentity(m_tracker.measurementNoiseCov, Scalar_<float>::all (1e-1f));
            setIdentity(m_tracker.errorCovPre, Scalar_<float>::all(1.0f));
            setIdentity(m_tracker.errorCovPost, Scalar_<float>::all(1.0f));
			// set initial state
            m_tracker.statePre.at<float>(0) = static_cast<float>(box.x);
            m_tracker.statePre.at<float>(1) = static_cast<float>(box.y);
            m_tracker.statePre.at<float>(2) = static_cast<float>(box.width);
            m_tracker.statePre.at<float>(3) = static_cast<float>(box.height);
            m_tracker.statePre.copyTo(m_tracker.statePost);
			
			Mat kalmanMeasure = m_tracker.measurementMatrix * m_tracker.predict();
			m_kalmanMeasure.x = static_cast<int>(kalmanMeasure.at<float>(0));
			m_kalmanMeasure.y = static_cast<int>(kalmanMeasure.at<float>(1));
			m_kalmanMeasure.width = static_cast<int>(kalmanMeasure.at<float>(2));
			m_kalmanMeasure.height = static_cast<int>(kalmanMeasure.at<float>(3));
			
			m_hasInit = true;
			m_hasMeasureUpdated = false;
		}

		void CKalmanFilter::Update(const cv::Rect& box)
		{
			if (!m_hasInit)
				return;

			Mat measure = Mat(MEASURE_D, 1, CV_32FC1);
			float* mptr = measure.ptr<float>();
			mptr[0] = static_cast<float>(box.x);
			mptr[1] = static_cast<float>(box.y);
			mptr[2] = static_cast<float>(box.width);
			mptr[3] = static_cast<float>(box.height);
			m_tracker.correct(measure);

			m_hasMeasureUpdated = false;
		}
		
		const cv::Rect& CKalmanFilter::Measure() const
		{
			if (m_hasMeasureUpdated)	// if !m_hasInit, the return will be a default initialised Mat
				return m_kalmanMeasure;

			Mat kalmanMeasure = m_tracker.measurementMatrix * m_tracker.predict();
			m_kalmanMeasure.x = static_cast<int>(kalmanMeasure.at<float>(0));
			m_kalmanMeasure.y = static_cast<int>(kalmanMeasure.at<float>(1));
			m_kalmanMeasure.width = std::max(static_cast<int>(kalmanMeasure.at<float>(2)), 0);
			m_kalmanMeasure.height = std::max(static_cast<int>(kalmanMeasure.at<float>(3)), 0);

			m_hasMeasureUpdated = true;
			return m_kalmanMeasure;
		}

		const cv::Mat& CKalmanFilter::StatePost() const
		{
			return m_tracker.statePost;
		}

		bool CKalmanFilter::HasMeasureUpdated() const
		{
			return m_hasMeasureUpdated;
		}

		void CKalmanFilter::Release()
		{
			m_tracker = KalmanFilter();
			m_kalmanMeasure = Rect();
			m_hasMeasureUpdated = false;
			m_hasInit = false;
		}

		/****************************************/
		// MeanShift tracker
		/****************************************/
		SMeanShiftParam::SMeanShiftParam(const std::string& configFile)
		{
			CREQUIRE(!configFile.empty(), "No configuration file specified");

			Json::Reader reader;
			Json::Value root;
			std::ifstream ifs(configFile.c_str());
			CREQUIRE(ifs.is_open(), "Error opening configuration file: " + configFile);
			CREQUIRE(reader.parse(ifs, root), "Error parsing configuration file: " + reader.getFormattedErrorMessages());

			const auto& config = root["Tracking"]["MeanShiftFilter"];

			m_numBins = config["NumberBins"].asInt();
			m_maxIter = config["MaxIteration"].asInt();
			m_histUpdateRate = config["HistogramUpdateRate"].asFloat();
			m_BHCoeffThreshold = config["BHCoeffThreshold"].asFloat();
		}

		std::unique_ptr<const SMeanShiftParam> CMeanShiftFilter::m_meanshiftParam(nullptr);
		void CMeanShiftFilter::SetGlobalConfig(const std::string& configFile)
		{
			m_meanshiftParam.reset(new SMeanShiftParam(configFile));
		}

		void CMeanShiftFilter::Init(const cv::Rect& box, const cv::Mat& frame, const cv::Mat* fg)
		{
			if (!m_meanshiftParam)
				throw CException("Global configuration parameters are not set");

			Mat kernel;
			CalcEpanechnikovKernel(box, &kernel);
			// calculate colour histogram
			CalcHistogram(box, kernel, frame, fg, m_meanshiftParam->m_numBins, &m_histogram, &m_histogramVol);
			m_meanshiftMeasure = box;
			m_BHCoeff = 1.0f;
			m_hasInit = true;
			m_hasMeasureUpdated = false;
		}

		void CMeanShiftFilter::CalcHistogram(const cv::Rect& box, const cv::Mat& kernel, const cv::Mat& frame, const cv::Mat* fg, int numBins, cv::Mat* hist, float* histVol)
		{
			// calculate colour histogram
			int totalNumBins = static_cast<int>(std::pow(numBins, frame.channels()));
			hist->create(1, totalNumBins, CV_32FC1);
			hist->setTo(0);
			*hist = *hist + (1.0f/totalNumBins);
			*histVol = 1;

			for (int y = 0; y < box.height; ++y)
			{
				int coordinateY = box.tl().y + y;
				if (!CoordinateInFrame(frame.size(), coordinateY, EAxis::Y))
					continue;

				for (int x = 0; x < box.width; ++x)
				{
					int coordinateX = box.tl().x + x;
					if (!CoordinateInFrame(frame.size(), coordinateX, EAxis::X))
						continue;

					Vec3b colour = frame.at<Vec3b>(coordinateY, coordinateX);
					int index = CalcHistogramIndex(colour, numBins);
					float kernelValue = kernel.at<float>(y, x);

					if (fg)
					{
						hist->at<float>(index) += kernelValue * (fg->at<uchar>(coordinateY, coordinateX) + 1);
						*histVol += kernelValue * (fg->at<uchar>(coordinateY, coordinateX) + 1);
					}
					else
					{
						hist->at<float>(index) += kernelValue;
						*histVol += kernelValue;
					}

				}
			}

		}

		float CMeanShiftFilter::BHCoeffThreshold()
		{ 
			if (m_meanshiftParam.get())
				return m_meanshiftParam->m_BHCoeffThreshold;
			else
				return 0.8;
		}

		void CMeanShiftFilter::CalcEpanechnikovKernel(const cv::Rect& box, cv::Mat* kernel)
		{
			kernel->create(box.height, box.width, CV_32FC1);
			kernel->setTo(0);
			Point centre = Point((box.height + 1) / 2, (box.width + 1) / 2);
			float denominator = (box.width * box.width + box.height * box.height) * 0.25f;
			for (int x = 0; x < box.height; ++x)
			{
				for (int y = 0; y < box.width; ++y)
				{
					float r2 = ((x - centre.x) * (x - centre.x) + (y - centre.y) * (y - centre.y)) / denominator;
					if (r2 < 1)
						kernel->at<float>(x,y) = 1 - r2;
				}
			}
		}

		void CMeanShiftFilter::CalcWeight(const cv::Rect& box, const cv::Mat& kernel, const cv::Mat& frame, const cv::Mat* fg, int numBins, const cv::Mat& hist, float histVol, cv::Mat* weight)
		{
			Mat candidateHist = hist;
			float candidateHistVol = histVol;
			weight->create(box.height, box.size().width, CV_32FC1);
			weight->setTo(0);
			float TH = 0.0f;
			float CH = 0.0f;
			for (int y = 0; y < box.height; ++y)
			{
				int coordinateY = box.tl().y + y;
				if (!CoordinateInFrame(frame.size(), coordinateY, EAxis::Y))
					continue;

				for (int x = 0; x < box.width; ++x)
				{
					int coordinateX = box.tl().x + x;
					if (!CoordinateInFrame(frame.size(), coordinateX, EAxis::X))
						continue;

					Vec3b colour = frame.at<Vec3b>(coordinateY, coordinateX);
					int index = CalcHistogramIndex(colour, numBins);

					if (m_histogramVol > std::numeric_limits<float>::epsilon())
						TH = m_histogram.at<float>(index)/m_histogramVol;
					if (candidateHistVol > std::numeric_limits<float>::epsilon())
						CH = candidateHist.at<float>(index)/candidateHistVol;

					if (CH > std::numeric_limits<float>::epsilon())
					{
						float tmpWeight = std::sqrt(TH/CH);
						if (fg && fg->at<uchar>(coordinateY, coordinateX))
							tmpWeight += 2;

						weight->at<float>(y, x) = std::min<float>(tmpWeight, 1e5);
					}

				}
			}
		}

		float CMeanShiftFilter::CalcBhattacharyya(const cv::Mat& targetHist, float targetHistVol, const cv::Mat& candidateHist, float candidateHistVol)
		{
			Mat tmp;
			cv::sqrt(targetHist.mul(candidateHist), tmp);
			return static_cast<float>(cv::sum(tmp)[0]) / std::sqrt(targetHistVol * candidateHistVol);
		}

		void CMeanShiftFilter::Update(const cv::Rect& box, const cv::Mat& frame, const cv::Mat* fg)
		{
			if (!m_hasInit)
				return;

			// prepare candidate information, predicted track's centre is used for MS
			Rect candidateBox = box;
			Point candidateCentre = BoxCentre(candidateBox);
			Mat candidateKernel;
			CalcEpanechnikovKernel(candidateBox, &candidateKernel);

			Mat candidateHist;
			float candidateHistVol = 0.0f;
			for (int iterCount = 0; iterCount < m_meanshiftParam->m_maxIter; ++iterCount)
			{
				float dx = 0.0f;
				float dy = 0.0f;
				float sumt = 0.0f;

				// meanshift update
				CalcHistogram(candidateBox, candidateKernel, frame, fg, m_meanshiftParam->m_numBins, &candidateHist, &candidateHistVol);
				Mat weight;
				CalcWeight(candidateBox, candidateKernel, frame, fg, m_meanshiftParam->m_numBins, candidateHist, candidateHistVol, &weight);

				for (int y = 0; y < candidateBox.height; ++y)
				{
					for (int x = 0; x < candidateBox.width; ++x)
					{
						float kw = candidateKernel.at<float>(y, x) * weight.at<float>(y, x);
						sumt += kw;
						dx += kw * (x - candidateBox.width/2);
						dy += kw * (y - candidateBox.height/2);
					}
				}

				if (sumt > std::numeric_limits<float>::epsilon())
				{
					dx /= sumt;
					dy /= sumt;
				}

				candidateBox.x += static_cast<int>(dx);
				candidateBox.y += static_cast<int>(dy);

				if (std::abs(dx) < 0.5 && std::abs(dy) < 0.5)
					break;
			}

			m_meanshiftMeasure = candidateBox;
			// calculate BhCoef for measuring tracking performance
			m_BHCoeff = CalcBhattacharyya(m_histogram, m_histogramVol, candidateHist, candidateHistVol);

			// update histogram information
			float vol = (candidateHistVol + m_histogramVol) * 0.5f;
			float currW = vol * m_meanshiftParam->m_histUpdateRate / candidateHistVol;
			float prevW = vol * (1 - m_meanshiftParam->m_histUpdateRate) / m_histogramVol;
			m_histogram = m_histogram * prevW + candidateHist * currW;
			m_histogramVol = static_cast<float>(cv::sum(m_histogram)[0]);

			m_hasMeasureUpdated = true;
		}

		const cv::Rect& CMeanShiftFilter::Measure() const
		{
			return m_meanshiftMeasure;
		}

		double CMeanShiftFilter::ConfidenceLevel() const
		{
			return m_BHCoeff;
		}

		bool CMeanShiftFilter::HasMeasureUpdated() const
		{
			return m_hasMeasureUpdated;
		}

		void CMeanShiftFilter::SetMeasureUpdated(bool value)
		{
			m_hasMeasureUpdated = false;
		}

		void CMeanShiftFilter::Release()
		{
			m_histogram.release();
			m_histogramVol = 0.0f;
			m_meanshiftMeasure = Rect();
			m_BHCoeff = 0.0f;
			m_hasMeasureUpdated = false;
			m_hasInit = false;
		}


		/****************************************/
		// tracker information
		/****************************************/
		void STrackerInfo::Init(const cv::Mat& frame, const cv::Mat* fg, int id, long startFrame, const cv::Rect& box, EObjectCategory objectType, int reservedTrajectoryLength)
        {
			InitKalmanOnly(id, startFrame, box, objectType, reservedTrajectoryLength);
			m_meanshift.Init(box, frame, fg);
        }

		void STrackerInfo::InitKalmanOnly(int id, long startFrame, const cv::Rect& box, EObjectCategory objectType, int reservedTrajectoryLength)
		{
            m_id                      = id;
            m_startFrame              = startFrame;
            m_duration                = 0;
            m_propagation             = 0;
			m_totalPropagation        = 0;
            m_validityScore           = 0.0f;
            m_type                    = objectType;
			m_status.AddStatus(SObjectStatus::Entering);
			m_kalman.Init(box);
			m_meanshift.Release();
			m_features.m_boundingBoxes.reserve(reservedTrajectoryLength);
			m_features.m_boundingBoxes.push_back(box);
			m_features.m_latestBoundingBox = box;
		}

		void STrackerInfo::Release()
		{
			m_id                 = 0;
			m_startFrame         = 0;
			m_duration           = 0;
			m_propagation        = 0;
			m_totalPropagation   = 0;
			m_validityScore      = 0.0f;
			m_type          = EObjectCategory::Unknown;
			m_kalman.Release();
			m_meanshift.Release();
			m_features.Release();
		}

		/****************************************/
		// data association parameters
		/****************************************/
		SDataAssociationParam::SDataAssociationParam(const std::string& configFile)
		{
			CREQUIRE(!configFile.empty(), "No configuration file specified");

			Json::Reader reader;
			Json::Value root;
			std::ifstream ifs(configFile.c_str());
			CREQUIRE(ifs.is_open(), "Error opening configuration file: " + configFile);
			CREQUIRE(reader.parse(ifs, root), "Error parsing configuration file: " + reader.getFormattedErrorMessages());

			const auto& config = root["Tracking"]["DataAssociation"];

			m_costAlpha = config["Alpha"].asFloat();
			m_costDistMax = config["MaxDistance"].asFloat();
			m_costType = config["CostType"].asString();
			m_algorithmType = config["AlgorithmType"].asString();
		}

		/****************************************/
		// miscellaneous parameters
		/****************************************/
		STrackerMiscellaneousness::STrackerMiscellaneousness(float objSizeUpdateRate, int minDispDuration, int stopDuration, int maxAllowedTracks, const cv::Size& minBox, const cv::Size& extension)
			: m_objSizeUpdateRate(objSizeUpdateRate)
			, m_minDispDuration(minDispDuration)
			, m_stopDuration(stopDuration)
			, m_maxAllowedTracks(maxAllowedTracks)
			, m_minBox(minBox)
			, m_extension(extension)
		{
		}

		STrackerMiscellaneousness::STrackerMiscellaneousness(const std::string& configFile)
		{
			CREQUIRE(!configFile.empty(), "No configuration file specified");

			Json::Reader reader;
			Json::Value root;
			std::ifstream ifs(configFile.c_str());
			CREQUIRE(ifs.is_open(), "Error opening configuration file: " + configFile);
			CREQUIRE(reader.parse(ifs, root), "Error parsing configuration file: " + reader.getFormattedErrorMessages());

			const auto& config = root["Tracking"]["Miscellaneousness"];
			m_objSizeUpdateRate = config["ObjectSizeUpdateRate"].asFloat();
			m_minDispDuration = config["MinDisplayDuration"].asInt();
			m_stopDuration = config["StopDuration"].asInt();
			m_maxAllowedTracks = config["MaxAllowedTracks"].asInt();
			m_minBox.height = config["MinBoxHeight"].asInt();
			m_minBox.width = config["MinBoxWidth"].asInt();
			m_extension.height = config["ExtensionHight"].asInt();
			m_extension.width = config["ExtensionWidth"].asInt();
		}

		/****************************************/
		// multiple object tracker
		/****************************************/
		CMultiTracker::CMultiTracker(const SDataAssociationParam& assocParam, const STrackerMiscellaneousness& miscellaneousness, const cv::Size& frameSize)
			: m_frameCount()
			, m_totalTrackCount()
			, m_totalRecordedTrackCount()
			, m_tracks()
			, m_dataAssocParam(assocParam)
			, m_miscellaneousness(miscellaneousness)
			, m_objectPool()
		{
		}

		CMultiTracker::CMultiTracker(const std::string& configFile)
			: m_frameCount()
			, m_totalTrackCount()
			, m_totalRecordedTrackCount()
			, m_tracks()
			, m_dataAssocParam(configFile)
			, m_miscellaneousness(configFile)
			, m_objectPool()
		{
			CKalmanFilter::SetGlobalConfig(configFile);
			CMeanShiftFilter::SetGlobalConfig(configFile);
		}

		void CMultiTracker::InitTrack(const cv::Rect& box)
		{
			STrackerInfo& newTrack = m_objectPool.AcquireObject();
			newTrack.InitKalmanOnly(m_totalTrackCount++, m_frameCount, box, EObjectCategory::Unknown, PRESERVED_TRAJECTORY_LENGTH);
			m_tracks.push_back(&newTrack);
		}

		void CMultiTracker::ReleaseTrack(STrackerInfo& obj)
		{
			obj.Release();
			m_objectPool.ReleaseObject(obj);
		}

		bool CMultiTracker::IsInvalidBox(const cv::Rect& box) const
		{
			return box.width < m_miscellaneousness.m_minBox.width || box.height < m_miscellaneousness.m_minBox.height ? true : false;
		}

		void CMultiTracker::CalcCostMatrix(const std::vector<cv::Rect>& boxes, std::vector<std::vector<float> >* costMatrix, std::vector<std::vector<int> >* gate) const
		{
			auto costDist = m_dataAssocParam.m_costDistMax;
			for (size_t i = 0; i < costMatrix->size(); ++i)
			{
				std::transform(boxes.begin(), boxes.end(), (*costMatrix)[i].begin(), SCostCalc(m_tracks[i]->m_kalman.Measure(), m_dataAssocParam.m_costAlpha));
				if (gate != nullptr)
					std::transform((*costMatrix)[i].begin(), (*costMatrix)[i].end(), (*gate)[i].begin(), [costDist](float value){ return value < costDist; });
			}
		}

		int CMultiTracker::FindBestMatchedObject(const cv::Rect& observation, const std::vector<int>& associationIndex) const
		{
			vector<int> objIndex;
			for (size_t i = 0; i < associationIndex.size(); ++i)
			{
				if (associationIndex[i] == 1)
					objIndex.push_back(i);
			}

			vector<float> scores;
			for (size_t i = 0; i < objIndex.size(); ++i)
			{
				float objBoxArea = static_cast<float>(m_tracks[objIndex[i]]->m_kalman.Measure().width * m_tracks[objIndex[i]]->m_kalman.Measure().height);
				float obsBoxArea = static_cast<float>(observation.width * observation.height);
				scores.push_back(abs(objBoxArea-obsBoxArea)/obsBoxArea);
			}

			vector<float>::iterator minIter = std::min_element(scores.begin(), scores.end());
			if (minIter != scores.end())
				return objIndex[std::distance(scores.begin(), minIter)];
			
			return -1;
		}

		void CMultiTracker::MergeSplittedObject(std::vector<cv::Rect>* observations, std::vector<std::vector<int> >* assocMatrix, const cv::Size& extension) const
		{
			int numObservations = observations->size();
			int numTracks = assocMatrix->size();

			vector<int> assocObsInd(numObservations, 0);
			for (int i = 0; i < numTracks; ++i)
			{
				for (int j = 0; j < numObservations; ++j)
					assocObsInd[j] += (*assocMatrix)[i][j];
			}
			
			vector<int> unassocObsInd;
			for (size_t i = 0; i < assocObsInd.size(); ++i)
			{
				if (assocObsInd[i] == 0)
					unassocObsInd.push_back(i);
			}

			vector<int> mergedObsInd(numObservations, 0);
			vector<int> reassocInd(numTracks, 0);
			for (size_t i = 0; i < unassocObsInd.size(); ++i)
			{
				for (int j = 0; j < numTracks; ++j)
				{
					const auto& unassocObsBox = (*observations)[unassocObsInd[i]];
					const auto& predictedTrackBox = m_tracks[j]->m_kalman.Measure();
					if (IsBoxesClosed(predictedTrackBox, unassocObsBox, extension))
					{
						auto index = std::distance((*assocMatrix)[j].begin(), std::find((*assocMatrix)[j].begin(), (*assocMatrix)[j].end(), 1));
						if (index == numObservations) // in case of track has no association
							reassocInd[j] = 1;
						else
						{
							const auto& obsBox = (*observations)[index];
							auto obsCentre = BoxCentre(obsBox);
							auto unassocObsCentre = BoxCentre(unassocObsBox);
							float dist = std::pow(static_cast<float>(obsCentre.x - unassocObsCentre.x), 2) + std::pow(static_cast<float>(obsCentre.y - unassocObsCentre.y), 2);
							float boxThreshold = std::pow(obsBox.width + unassocObsBox.width * 0.5f + extension.width, 2) +
												 std::pow(obsBox.height + unassocObsBox.height * 0.5f + extension.height, 2);
							if (dist < std::max(m_dataAssocParam.m_costDistMax, boxThreshold))
							{
								mergedObsInd[unassocObsInd[i]] = 1;
								(*observations)[index] = MergeBoxesImpl((*observations)[index], (*observations)[unassocObsInd[i]]);
							}
						}
					}
				}

				if (std::any_of(reassocInd.begin(), reassocInd.end(), [](int value) { return value == 1; }))
				{
					// find the best match for the track without observation
					int index = FindBestMatchedObject((*observations)[unassocObsInd[i]], reassocInd);
					if (index >= 0)
						(*assocMatrix)[index][unassocObsInd[i]] = 1;	// add new association in the matrix
				}
			}

			// delete merged observations in the observation list
			int count = 0;
			observations->erase(std::remove_if(observations->begin(), observations->end(), [&mergedObsInd, &count](const Rect& observation) mutable { return mergedObsInd[count++] == 1; }), observations->end());

			vector<vector<int> > refinedAssocMatrix;
			for (int i = 0; i < numTracks; ++i)
			{
				vector<int> rowAssoc;
				int count = 0;
				std::copy_if((*assocMatrix)[i].begin(), (*assocMatrix)[i].end(), std::back_inserter(rowAssoc), [&mergedObsInd, count](int elem) mutable { return mergedObsInd[count++] == 0; });
				refinedAssocMatrix.push_back(std::move(rowAssoc));
			}

			refinedAssocMatrix.swap(*assocMatrix);
		}

		void CMultiTracker::CalcStatusMatrix(const std::vector<cv::Rect>& observations, std::vector<std::vector<int> >* assocMatrix, std::vector<std::vector<int> >* statusMatrix, const cv::Size& extension) const
		{
			vector<int> assocTracksInd(assocMatrix->size(), 0);
			std::transform(assocMatrix->begin(), assocMatrix->end(), assocTracksInd.begin(), [](const vector<int>& rowAssoc){ return std::find(rowAssoc.begin(), rowAssoc.end(), 1) != rowAssoc.end(); });

			vector<int> assocObsInd(observations.size(), 0);
			for (size_t i = 0; i < assocMatrix->size(); ++i)
			{
				for (size_t j = 0; j < observations.size(); ++j)
				{
					assocObsInd[j] += (*assocMatrix)[i][j];
				}
			}

			// originally, we only use predicted objects box position and observation FG
			// to decide if two tracks are overlapped, this is not sufficient enough.
			// Because the remaining shadow after shadow removing in the FG detection
			// part may occasionally join two closed tracks (which outside the range of
			// extension), then in the loop, we can not confirm the two tracks are
			// closed, further more, no overlapping. In order to improve the detection,
			// we should also consider the observation FG. If the FG overlapped with both
			// tracks, even if the two tracks are not closed, they are still overlapped.
			for (size_t i = 0; i < assocMatrix->size(); ++i)
			{
				if (!m_tracks[i]->m_status.HasStatus(SObjectStatus::Entering))	// for unstable detection, we don't count its overlapping case
				{
					for (size_t j = 0; j < assocMatrix->size(); ++j)
					{
						if (i != j && !m_tracks[j]->m_status.HasStatus(SObjectStatus::Entering))
						{
							bool isClosed = IsBoxesClosed(m_tracks[i]->m_kalman.Measure(), m_tracks[j]->m_kalman.Measure(), extension);
							if (isClosed && assocTracksInd[i] && assocTracksInd[j])
								(*statusMatrix)[i][j] = IS_CLOSED;
							else if (isClosed && (assocTracksInd[i] || assocTracksInd[j]) && !(assocTracksInd[i] && assocTracksInd[j]))
								(*statusMatrix)[i][j] = IS_OVERLAPPED;
						}
					}
				}
			}

			for (size_t i = 0; i < assocMatrix->size(); ++i)
			{
				if (!m_tracks[i]->m_status.HasStatus(SObjectStatus::Entering))
				{
					for (size_t j = 0; j < observations.size(); ++j)
					{
						if ((*assocMatrix)[i][j] == 0)
						{
							if (IsBoxesClosed(m_tracks[i]->m_kalman.Measure(), observations[j], Size(0, 0)))
							{
								if (assocObsInd[j])
								{
									// track should has association, find associated track
									int trackInd = -1;
									for (size_t k = 0; k < assocMatrix->size(); ++k)
									{
										if ((*assocMatrix)[k][j] == 1)
										{
											trackInd = k;
											break;
										}
									}

									// mark the tracks as overlapped
									if (trackInd >= 0 && (*assocMatrix)[trackInd][j] == 0)
									{
										(*statusMatrix)[i][trackInd] = IS_OVERLAPPED;
										(*statusMatrix)[trackInd][i] = IS_OVERLAPPED;
									}
								}
								else
								{
									if (assocTracksInd[i] == 0)
									{
										(*statusMatrix)[i][j] = IS_OVERLAPPED;
										(*statusMatrix)[j][i] = IS_OVERLAPPED;
									}
								}
							}
						}
					}
				}
			}
		}

		std::vector<int> CMultiTracker::FindOverlappedObjectsLayers(const std::vector<STrackerInfo*>& trackerInfo) const
		{
			if (trackerInfo.empty())
				return vector<int>();

			// assuming track with largest br() is at the front layer, all the others are at the back layer
			int yMax = 0;
			int topLayerIndex = 0;
			for (size_t i = 0; i < trackerInfo.size(); ++i)
			{
				if (trackerInfo[i]->m_kalman.Measure().br().y > yMax)
				{
					yMax = trackerInfo[i]->m_kalman.Measure().br().y;
					topLayerIndex = i;
				}
			}
			vector<int> layers(trackerInfo.size(), IS_BACK_LAYER);
			layers[topLayerIndex] = IS_FRONT_LAYER;

			return layers;
		}

		bool CMultiTracker::IsObjectStopping(const std::vector<cv::Rect>& boxes) const
		{
			int length = std::min(static_cast<int>(boxes.size()), m_miscellaneousness.m_stopDuration);
			float averageCentreX = 0.0f;
			float averageCentreY = 0.0f;
			float averageWidth = 0.0f;
			float averageHeight = 0.0f;
			vector<Point> centres;
			for (int i = boxes.size() - length; i < static_cast<int>(boxes.size()); ++i)
			{
				Point centre = BoxCentre(boxes[i]);
				averageCentreX += centre.x;
				averageCentreY += centre.y;
				averageWidth += boxes[i].width;
				averageHeight += boxes[i].height;
				centres.push_back(std::move(centre));
			}
			averageCentreX /= length;
			averageCentreY /= length;
			averageWidth /= length;
			averageHeight /= length;

			float varCentreX = 0.0f;
			float varCentreY = 0.0f;
			for (size_t i = 0; i < centres.size(); ++i)
			{
				varCentreX += std::pow(centres[i].x - averageCentreX, 2);
				varCentreY += std::pow(centres[i].y - averageCentreY, 2);
			}
			varCentreX /= (length - 1);
			varCentreY /= (length - 1);

			if (varCentreX < averageWidth * averageWidth * 0.0625 || varCentreY < averageHeight * averageHeight * 0.0625)
				return true;
			
			return false;

		}

		void CMultiTracker::UpdateTracks(const std::vector<cv::Rect>& boxes, const cv::Mat& frame, const cv::Mat* fg, const std::vector<cv::Rect>* definedEntries)
		{
			// merge nearby foreground boxes
			auto observations = MergeBoxes(boxes, m_miscellaneousness.m_extension);
			// removed small boxes
			observations.erase(std::remove_if(observations.begin(), observations.end(), boost::bind(&CMultiTracker::IsInvalidBox, this, _1)), observations.end());
			 
			if (static_cast<int>(observations.size()) - static_cast<int>(m_tracks.size()) > m_miscellaneousness.m_maxAllowedTracks)
				observations.clear();	// too many observations, maybe outliers

			vector<int> rowAssoc(observations.size(), 0);
			vector<vector<int> > assocMatrix(m_tracks.size(), rowAssoc);
			int newTrackCount = 0;
			if (m_tracks.size() == 0 && observations.size() != 0)
			{
				for (const auto& box: observations)
				{
					InitTrack(box);
					newTrackCount++;
				}
			}
			else if (m_tracks.size() != 0)
			{
				// kalman prediction
				for (auto& track: m_tracks)
				{
					track->m_kalman.Measure();
					if (track->m_meanshift.HasInit())
						track->m_meanshift.SetMeasureUpdated(false);
				}

				// data association
				// if no observations are found in the current frame, every unexited tracks will be 
				// propagated using its previous information in kalman filter
				vector<float> rowCost(observations.size(), 0.0f);
				vector<vector<float> > costMatrix(m_tracks.size(), rowCost);
				vector<int> rowStatus(m_tracks.size(), 0);
				vector<vector<int> > statusMatrix(m_tracks.size(), rowStatus);

				if (!observations.empty())
				{
					// initialise cost matrix
					CalcCostMatrix(observations, &costMatrix);

					// initialise association matrix (numTracks x numMergedBoxes)
					LinearAssignment(costMatrix, m_dataAssocParam.m_costDistMax, assocMatrix);

					// to avoid object split into several small parts caused by false detection.
					// we place previous object box in the predicted position, all small
					// unassociated FGs within and near it will be merged to a single object
					MergeSplittedObject(&observations, &assocMatrix, Size(0, 0));

					CalcStatusMatrix(observations, &assocMatrix, &statusMatrix);

					// add new tracks for unassinged observations
					for (size_t i = 0; i < observations.size(); ++i)
					{
						bool isAssigned = std::any_of(assocMatrix.begin(), assocMatrix.end(), [i](const vector<int>& rowAssoc){ return rowAssoc[i] == 0 ? false : true; });
						if (!isAssigned)
						{
							InitTrack(observations[i]);
							newTrackCount++;
						}
					}
				}

				/*************explicit tracking update begin*************/
				//auto assocMatrixBackup = assocMatrix;

				vector<int> assocTracksInd;
				for (size_t i = 0; i < assocMatrix.size(); ++i)
				{
					if (std::any_of(assocMatrix[i].begin(), assocMatrix[i].end(), [](int i){ return i == 1; }))
						assocTracksInd.push_back(i);
				}				

				if (!assocTracksInd.empty())
				{
					for (size_t i = 0; i < assocTracksInd.size(); ++i)
					{
						int trackInd = assocTracksInd[i];
						auto track = m_tracks[trackInd];
						int obsInd = std::distance(assocMatrix[trackInd].begin(), std::find(assocMatrix[trackInd].begin(), assocMatrix[trackInd].end(), 1));

						// find all overlapped tracks associated with trackInd
						vector<int> overlappedTracksInd;
						for (size_t j = 0; j < statusMatrix[0].size(); ++j)
						{
							if (statusMatrix[trackInd][j] == IS_OVERLAPPED)
								overlappedTracksInd.push_back(j);
						}

						// only do meanshift for non-entering tracks, as the size of tracks in the entering changes significantly.
						if (!track->m_status.HasStatus(SObjectStatus::Entering/* | SObjectStatus::Exiting*/))
						{
							if (overlappedTracksInd.empty()) // track has no overlapped track
							{
								if (!track->m_meanshift.HasInit())
									track->m_meanshift.Init(observations[obsInd], frame, fg);
								else
									track->m_meanshift.Update(observations[obsInd], frame, fg);
							}
							else // track has overlapped tracks
							{
								// find the layers of all overlapped tracks
								vector<STrackerInfo*> tmpTrackerInfo;
								tmpTrackerInfo.push_back(track);

								track->m_status.AddStatus(SObjectStatus::Overlapping);
								for (size_t k = 0; k < overlappedTracksInd.size(); ++k)
								{
									tmpTrackerInfo.push_back(m_tracks[overlappedTracksInd[k]]);
									m_tracks[overlappedTracksInd[k]]->m_status.AddStatus(SObjectStatus::Overlapping);
								}

								auto layerInfo = FindOverlappedObjectsLayers(tmpTrackerInfo);
								int frontLayerIndex = std::distance(layerInfo.begin(), std::find(layerInfo.begin(), layerInfo.end(), IS_FRONT_LAYER));
								bool IsFrontLayerMeanShiftInit = tmpTrackerInfo[frontLayerIndex]->m_meanshift.HasInit();
								bool noMeanShiftTracking = false;
								if (IsFrontLayerMeanShiftInit)
								{
									for (size_t k = 0; k < layerInfo.size(); ++k)
									{
										if (k != frontLayerIndex && tmpTrackerInfo[k]->m_meanshift.HasInit())
										{
											auto BHValue = CMeanShiftFilter::CalcBhattacharyya(tmpTrackerInfo[frontLayerIndex]->m_meanshift.Histogram(), tmpTrackerInfo[frontLayerIndex]->m_meanshift.HistogramVolumn(), tmpTrackerInfo[k]->m_meanshift.Histogram(), tmpTrackerInfo[k]->m_meanshift.HistogramVolumn());
											if (BHValue > CMeanShiftFilter::BHCoeffThreshold())  // overlapped tracks colour are similar, disable meanshift temporally
												noMeanShiftTracking = true;
										}
									}

									if (!noMeanShiftTracking)
										tmpTrackerInfo[frontLayerIndex]->m_meanshift.Update(tmpTrackerInfo[frontLayerIndex]->m_kalman.Measure(), frame, fg);
								}
							}
						}

						if (overlappedTracksInd.empty() && track->m_status.HasStatus(SObjectStatus::Overlapping))
							track->m_status.RemoveStatus(SObjectStatus::Overlapping);
						else if (track->m_status.HasStatus(SObjectStatus::Lost))
							track->m_status.RemoveStatus(SObjectStatus::Lost);
					}
				}
				for (size_t i = 0; i < m_tracks.size() - newTrackCount; ++i)
				{
					if (std::find(assocTracksInd.begin(), assocTracksInd.end(), i) == assocTracksInd.end() && !m_tracks[i]->m_status.HasStatus(SObjectStatus::Overlapping))
					{
						m_tracks[i]->m_status.AddStatus(SObjectStatus::Lost);

						if (m_tracks[i]->m_meanshift.HasInit() && !m_tracks[i]->m_status.HasStatus(SObjectStatus::Exiting) && IsObjectStopping(m_tracks[i]->m_features.m_boundingBoxes))
						{
							Rect newMeasure(m_tracks[i]->m_kalman.Measure().x, m_tracks[i]->m_kalman.Measure().y, m_tracks[i]->m_features.m_latestBoundingBox.width, m_tracks[i]->m_features.m_latestBoundingBox.height);
							m_tracks[i]->m_meanshift.Update(newMeasure, frame, fg);
						}
						else if (m_tracks[i]->m_status.HasStatus(SObjectStatus::Exiting))
							m_tracks[i]->m_removable = true;
						
					}
				}
			}

			// update and store tracks bounding box at different status
			for (size_t i = 0; i < m_tracks.size() - newTrackCount; ++i) // for existing tracks
			{
				// find the foreground box
				int index = std::distance(assocMatrix[i].begin(), std::find(assocMatrix[i].begin(), assocMatrix[i].end(), 1));
				const Rect* box = nullptr;
				if (index != assocMatrix[i].size())
					box = &observations[index];

				auto& track = m_tracks[i];
				if (box && track->m_meanshift.HasMeasureUpdated())
				{
					if (track->m_status.HasStatus(SObjectStatus::Overlapping))	// do not update size
						track->m_features.m_latestBoundingBox = track->m_meanshift.Measure();
					else
					{
						Point centre = BoxCentre(track->m_meanshift.Measure());
						int height = static_cast<int>(track->m_features.m_latestBoundingBox.height * (1 - m_miscellaneousness.m_objSizeUpdateRate) + box->height * m_miscellaneousness.m_objSizeUpdateRate);
						int width = static_cast<int>(track->m_features.m_latestBoundingBox.width * (1 - m_miscellaneousness.m_objSizeUpdateRate) + box->width * m_miscellaneousness.m_objSizeUpdateRate);
						track->m_features.m_latestBoundingBox = Rect(centre.x - width / 2, centre.y - height / 2, width, height);
					}
				}
				else if(box && !track->m_meanshift.HasMeasureUpdated())
				{
					Point centre = BoxCentre(*box);
					int height = static_cast<int>(track->m_features.m_latestBoundingBox.height * (1 - m_miscellaneousness.m_objSizeUpdateRate) + box->height * m_miscellaneousness.m_objSizeUpdateRate);
					int width = static_cast<int>(track->m_features.m_latestBoundingBox.width * (1 - m_miscellaneousness.m_objSizeUpdateRate) + box->width * m_miscellaneousness.m_objSizeUpdateRate);
					track->m_features.m_latestBoundingBox = Rect(centre.x - width / 2, centre.y - height / 2, width, height);
				}
				else if (!box && track->m_meanshift.HasMeasureUpdated())
				{
					track->m_features.m_latestBoundingBox = track->m_meanshift.Measure();
				}
				else
				{
					track->m_features.m_latestBoundingBox = track->m_kalman.Measure();
				}
				
				track->m_kalman.Update(track->m_features.m_latestBoundingBox);
			}

			for (size_t i = m_tracks.size() - newTrackCount; i < m_tracks.size(); ++i)
			{
				auto& track = m_tracks[i];
				track->m_features.m_latestBoundingBox = track->m_kalman.Measure();
			}


			for (size_t i = 0; i < m_tracks.size(); ++i)
			{
				auto& track = m_tracks[i];
				track->m_features.m_boundingBoxes.push_back(track->m_features.m_latestBoundingBox);
				track->m_duration++;
				if (track->m_status.HasStatus(SObjectStatus::Overlapping | SObjectStatus::Lost))
				{
					track->m_propagation++;
					track->m_totalPropagation++;
				}
				else
				{
					track->m_propagation = 0;
				}

				track->m_validityScore = 1 - static_cast<float>(track->m_totalPropagation) / track->m_duration;
					
			}

			////////////////////////////////
			// TODO: HLI handle object split
			////////////////////////////////

			////////////////////////////////
			// TODO: HLI handle object merge
			////////////////////////////////


			
			for (auto& track: m_tracks)
			{
				if (track->m_duration > m_miscellaneousness.m_minDispDuration)
				{
					track->m_isRecording = true;
					vector<Rect> boxes(track->m_features.m_boundingBoxes.end() - std::min(static_cast<int>(track->m_features.m_boundingBoxes.size()), 3), track->m_features.m_boundingBoxes.end());
					if (!track->m_status.HasStatus(SObjectStatus::In) && !track->m_status.HasStatus(SObjectStatus::Overlapping) && 
						!IsBoxesCloseToEntries(boxes, frame.size(), nullptr))
					{
						track->m_status.AddStatus(SObjectStatus::In);
						track->m_status.RemoveStatus(SObjectStatus::Entering | SObjectStatus::Exiting);
					}
					else if (track->m_status.HasStatus(SObjectStatus::In) && 
							 IsBoxesCloseToEntries(boxes, frame.size(), definedEntries))
					{
						track->m_status.AddStatus(SObjectStatus::Exiting);
						track->m_status.RemoveStatus(SObjectStatus::Entering | SObjectStatus::In);
					}
				}
				
				// TODO: HLI how to assign stopping status

			}

			// managing the track list
			for (auto & track: m_tracks)
			{
				if (!track->m_removable && 
					((track->m_isRecording && track->m_status.HasStatus(SObjectStatus::Overlapping) && track->m_propagation > CKalmanFilter::MaxOverlapPropagation()) || 
					(track->m_isRecording && track->m_status.HasStatus(SObjectStatus::Lost) && track->m_propagation > CKalmanFilter::MaxLostPropagation()) ||
					(!track->m_isRecording && track->m_propagation > CKalmanFilter::MaxProgrationNoRecord())))
				{
					ReleaseTrack(*track);
					track->m_removable = true;
					continue;
				}

				Point centre = BoxCentre(track->m_features.m_latestBoundingBox);
				if (!track->m_removable && (!CoordinateInFrame(frame.size(), centre.x, EAxis::X) || !CoordinateInFrame(frame.size(), centre.y, EAxis::Y)))
				{
					ReleaseTrack(*track);
					track->m_removable = true;
					continue;
				}

				if (!track->m_removable && (track->m_kalman.Measure().height == 0 || track->m_kalman.Measure().width == 0))
				{
					ReleaseTrack(*track);
					track->m_removable = true;
					continue;
				}
			}

			m_tracks.erase(std::remove_if(m_tracks.begin(), m_tracks.end(), [](const STrackerInfo* trackInfo) { return trackInfo->m_removable; }), m_tracks.end());

			m_frameCount++;
		}


		void CMultiTracker::DisplayTracks(cv::Mat& frame, const cv::Size& extension)
		{
			for (auto& track: m_tracks)
			{
				if (track->m_duration > m_miscellaneousness.m_minDispDuration && track->m_validityScore > 0.6)
				{
					string display;
					const Rect& box = track->m_features.m_latestBoundingBox;
					Point centre = BoxCentre(box);
					int width = box.width + 2 * extension.width;
					int height = box.height + 2 * extension.width;
					Rect displayBox = Rect(centre.x - width / 2, centre.y - height /2, width, height);

					putText(frame, display, displayBox.tl() , FONT_HERSHEY_PLAIN, 1, CVROI::Colour(CVROI::EColour::GREEN));
					rectangle(frame, displayBox.tl(), displayBox.br(), CVROI::Colour(CVROI::EColour::GREEN), 1);

					// draw most recent trajectory
					int startIndex = track->m_duration >= 50 ? track->m_duration - 50 : 0;
					for (int i = startIndex; i < track->m_duration - 1; ++i)
					{
						line(frame, BoxCentre(track->m_features.m_boundingBoxes[i]), BoxCentre(track->m_features.m_boundingBoxes[i+1]), CVROI::Colour(CVROI::EColour::GREEN)); 
					}
				}
			}
		}


		//void CMultiTracker::WriteToXML() const
		//{
		//	
		//}

    }
}