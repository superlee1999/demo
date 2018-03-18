#ifndef TRACKING_TRACKER_H
#define TRACKING_TRACKER_H
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <stdexcept>
#include <memory>
#include <math.h>
#include <cstdint>
#include "Tracking/Features.h"
#include "Utils/ObjectPool.h"
#include "Utils/Export.h"
#include <boost/noncopyable.hpp>

namespace Tracking {
    namespace Tracker {
		APT_API bool IsBoxCloseToEntries(const cv::Rect& box, const cv::Size frameSize, const std::vector<cv::Rect>* definedEntries);
		APT_API bool IsBoxesCloseToEntries(const std::vector<cv::Rect>& boxes, const cv::Size frameSize, const std::vector<cv::Rect>* definedEntries);
		/****************************************/
		// Kalman tracker
		/****************************************/
		struct APT_API SKalmanParam
        {
            SKalmanParam(int maxPro)
				: m_maxOverlapPropagate(maxPro)
				, m_maxLostPropagate(maxPro)
				, m_maxPropagateNoRecord(15)
			{}
			explicit SKalmanParam(const std::string& configFile);

            int m_maxOverlapPropagate;
			int m_maxLostPropagate;
			int m_maxPropagateNoRecord;
        };

		class APT_API CKalmanFilter
		{
		public:
			static void SetGlobalConfig(const std::string& configFile);	// only need to call once
			void Init(const cv::Rect& box);
			void Update(const cv::Rect& box);
			const cv::Rect& Measure() const;
			const cv::Mat& StatePost() const;
			bool HasMeasureUpdated() const;
			bool HasInit() const { return m_hasInit; }
			void Release();

			static int MaxOverlapPropagation() { return m_kalmanParam->m_maxOverlapPropagate; }
			static int MaxLostPropagation() { return m_kalmanParam->m_maxLostPropagate; }
			static int MaxProgrationNoRecord() { return m_kalmanParam->m_maxPropagateNoRecord; }

		private:
			enum EStatus {MEASURE_D = 4, STATE_D = 8};
			mutable cv::KalmanFilter m_tracker;
			mutable cv::Rect m_kalmanMeasure;	// predicted object state, kalman.measurementMatrix * kalman.predict()
			mutable bool m_hasMeasureUpdated;
			bool m_hasInit;
			static std::unique_ptr<const SKalmanParam> m_kalmanParam;
		};


		/****************************************/
		// MeanShift tracker
		/****************************************/
		struct APT_API SMeanShiftParam
		{
			SMeanShiftParam(int numBins, int maxIter, float histUpdateRate = 0.1, float objSizeUpdateRate = 0.1, float BHCoeffThreshold = 0.8)
				: m_numBins(numBins)
				, m_maxIter(maxIter)
				, m_histUpdateRate(histUpdateRate)
				, m_BHCoeffThreshold(BHCoeffThreshold)
			{
			}
			explicit SMeanShiftParam(const std::string& configFile);

			int m_numBins;
			int m_maxIter;
			float m_histUpdateRate;
			float m_BHCoeffThreshold;
		};

		class APT_API CMeanShiftFilter
		{
		public:
			static void SetGlobalConfig(const std::string& configFile);	// only need to call once
			void Init(const cv::Rect& box, const cv::Mat& frame, const cv::Mat* fg = nullptr);
			void Update(const cv::Rect& box, const cv::Mat& frame, const cv::Mat* fg = nullptr);
			const cv::Rect& Measure() const;
			double ConfidenceLevel() const;
			bool HasMeasureUpdated() const;
			void SetMeasureUpdated(bool value);
			bool HasInit() const { return m_hasInit; }
			void Release();

			const cv::Mat& Histogram() const { return m_histogram; }
			float HistogramVolumn() const { return m_histogramVol; }

			static float BHCoeffThreshold();
			static void CalcEpanechnikovKernel(const cv::Rect& box, cv::Mat* kernel);
			static void CalcHistogram(const cv::Rect& box, const cv::Mat& kernel, const cv::Mat& frame, const cv::Mat* fg, int numBins, cv::Mat* hist, float* histVol);
			static float CalcBhattacharyya(const cv::Mat& targetHist, float targetHistVol, const cv::Mat& candidateHist, float candidateHistVol);

		private:
			void CalcWeight(const cv::Rect& box, const cv::Mat& kernel, const cv::Mat& frame, const cv::Mat* fg, int numBins, const cv::Mat& hist, float histVol, cv::Mat* weight);

		private:
			cv::Mat m_histogram;
			float m_histogramVol;
			cv::Rect m_meanshiftMeasure;	// predicted object state
			float m_BHCoeff;
			bool m_hasMeasureUpdated;
			bool m_hasInit;
			static std::unique_ptr<const SMeanShiftParam> m_meanshiftParam;
		};


		/****************************************/
		// tracker information
		/****************************************/
        // hold per object information
		enum class EObjectCategory { Unknown, Human, Vehicle, Others };
		struct APT_API SObjectStatus
		{
			enum EStatus{
				Unknown     = 0x00,
				Entering    = 0x01,
				Exiting     = 0x02,
				In		    = 0x04,
				Stopping    = 0x08,
				Overlapping = 0x10,
				Lost		= 0x20,
				Splitting   = 0x40,
				Merging     = 0x80,
				Deleting    = 0x100,
				Falling		= 0x200,
				Running		= 0x400,
				Loitering	= 0x800
			};

			SObjectStatus() : m_status(Unknown) {}

			void AddStatus(int status) { m_status |= status; }
			void SetStatus(int status) { m_status = status; }
			void RemoveStatus(int status) { m_status &= ~status; }
			bool HasStatus(int status) const { return (m_status & status) != 0; }
			void RemoveAllStatus(int status) { m_status = 0; }

		private:
			uint32_t m_status; 
		};



        struct APT_API STrackerInfo
        {
            STrackerInfo()
				: m_id(0)
				, m_recordedId(0)
				, m_startFrame(0)
				, m_duration(0)
				, m_propagation(0)
				, m_totalPropagation(0)
				, m_validityScore(0.0f)
				, m_isRecording(false)
				, m_removable(false)
				, m_status()
				, m_type(EObjectCategory::Unknown)
				, m_kalman()
				, m_meanshift()
				, m_features() 
			{}

			void Init(const cv::Mat& frame, const cv::Mat* fg, int id, long startFrame, const cv::Rect& box, EObjectCategory objectType = EObjectCategory::Unknown, int reservedTrajectoryLength = 100);
			void InitKalmanOnly(int id, long startFrame, const cv::Rect& box, EObjectCategory objectType = EObjectCategory::Unknown, int reservedTrajectoryLength = 100);

            void Release();
			
            int                 m_id;	// a unique vlue
			int					m_recordedId; // actual id in the recorder
            long                m_startFrame;
            int                 m_duration;
            int                 m_propagation;
			int					m_totalPropagation;
            float               m_validityScore;
			bool				m_isRecording;
			bool				m_removable;
            EObjectCategory     m_type;
			SObjectStatus		m_status;
			CKalmanFilter		m_kalman;
			CMeanShiftFilter	m_meanshift;
            Features::SObjectFeatures      m_features;
        };

		
		/****************************************/
		// data association parameters
		/****************************************/
        struct SDataAssociationParam
        {
			SDataAssociationParam()
				: m_costAlpha()
				, m_costDistMax()
				, m_costType()
				, m_algorithmType()
			{
			}

            SDataAssociationParam(float alpha, float maxDist, std::string costtype = "FixedDistance", std::string algorihtmtype = "LAP")
				: m_costAlpha (alpha)
				, m_costDistMax(maxDist)
				, m_costType(costtype)
				, m_algorithmType(algorihtmtype)
            {}

			SDataAssociationParam(const std::string& configFile);

            float           m_costAlpha;
            float           m_costDistMax;
            std::string     m_costType;
            std::string     m_algorithmType;
        };


		/****************************************/
		// miscellaneous parameters
		/****************************************/
		struct STrackerMiscellaneousness
		{
			STrackerMiscellaneousness() {}
			STrackerMiscellaneousness(float objSizeUpdateRate, int minDispDuration, int stopDuration, int maxAllowedTracks, const cv::Size& minBox, const cv::Size& extension = cv::Size(5, 5));
			STrackerMiscellaneousness(const std::string& configFile);

			float m_objSizeUpdateRate;
			int m_minDispDuration;
			int m_stopDuration;
			int m_maxAllowedTracks;	// max number of fine tracks displayable on screen
			cv::Size m_minBox;
			cv::Size m_extension;
		};


        // common tracker API definition
        class APT_API CTrackerBase
        {
        public:
            CTrackerBase() {}
            virtual ~CTrackerBase() = 0 {}
            virtual void UpdateTracks(const std::vector<cv::Rect>& box, const cv::Mat& frame, const cv::Mat* fg, const std::vector<cv::Rect>* definedEntries) = 0;
            virtual void DisplayTracks(cv::Mat& frame, const cv::Size& extension = cv::Size(0, 0)) = 0;
			virtual const std::vector<STrackerInfo*> GetCurrentTracks(void) const = 0;

		protected:
			virtual void InitTrack(const cv::Rect& box) = 0;
            virtual void ReleaseTrack(STrackerInfo& obj) = 0;
        };


		class APT_API CMultiTracker : public CTrackerBase, private boost::noncopyable
		{
		public:
			CMultiTracker(const SDataAssociationParam& assocParam, const STrackerMiscellaneousness& miscellaneousness, const cv::Size& frameSize);
			CMultiTracker(const std::string& configFile);

            virtual void UpdateTracks(const std::vector<cv::Rect>& box, const cv::Mat& frame, const cv::Mat* fg, const std::vector<cv::Rect>* definedEntries = nullptr);
            virtual void DisplayTracks(cv::Mat& frame, const cv::Size& extension);
			//virtual void WriteToXML() const;
			virtual const std::vector<STrackerInfo*> GetCurrentTracks(void) const { return m_tracks; }
			std::vector<STrackerInfo*> GetCurrentTracks(void) { return m_tracks; }

		private:
			virtual void InitTrack(const cv::Rect& box);
            virtual void ReleaseTrack(STrackerInfo& obj);	// return the memory to object pool
			bool IsInvalidBox(const cv::Rect& box) const;
			void CalcCostMatrix(const std::vector<cv::Rect>& boxes, std::vector<std::vector<float> >* costMatrix, std::vector<std::vector<int> >* gate = nullptr) const;
			void MergeSplittedObject(std::vector<cv::Rect>* observations, std::vector<std::vector<int> >* assocMatrix, const cv::Size& extension = cv::Size(5,5)) const;
			int FindBestMatchedObject(const cv::Rect& observation, const std::vector<int>& associationIndex) const;
			void CalcStatusMatrix(const std::vector<cv::Rect>& observations, std::vector<std::vector<int> >* assocMatrix, std::vector<std::vector<int> >* statusMatrix, const cv::Size& extension = cv::Size(5, 5)) const;
			std::vector<int> FindOverlappedObjectsLayers(const std::vector<STrackerInfo*>& trackerInfo) const;
			bool IsObjectStopping(const std::vector<cv::Rect>& boxes) const;

		private:
			enum { IS_CLOSED = 1, IS_OVERLAPPED = 2, IS_FRONT_LAYER = 3, IS_BACK_LAYER = 4, PRESERVED_TRAJECTORY_LENGTH = 500 };

			long m_frameCount;
            int m_totalTrackCount;			// total number of tracked object
			int m_totalRecordedTrackCount;	// total number of objects actually recorded (fine track)
			std::vector<STrackerInfo*> m_tracks;
			SDataAssociationParam m_dataAssocParam;
			STrackerMiscellaneousness m_miscellaneousness;
			Utils::CObjectPool<STrackerInfo> m_objectPool;	// object pool for memory management

		//private:
		//	CMultiTracker(const CMultiTracker&);
		//	CMultiTracker& operator=(const CMultiTracker&);
		};
		
    }
}


#endif