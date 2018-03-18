#include "Tracking/Features.h"
#include <numeric>
#include <algorithm>
#include <math.h>

using namespace std;
using namespace cv;

namespace Tracking {
	namespace Features {
		/// COpticalFlow implementation details
		class COpticalFlow::Impl
		{
		public:
			class VectorDiff
			{
			public:
				Point2f operator()(const Point2f& p1, const Point2f& p2) const
				{
					return Point2f(p1.x-p2.x, p1.y-p2.y);
				}
			};

			class SqrtMagnitude
			{
			public:    
				float operator()(const Point2f& p) const
				{
					return sqrt(p.x*p.x + p.y*p.y);
				}
			};

			class ATan2
			{
			public:
				float operator()(const Point2f& p) const
				{
					return std::atan2 (p.y, p.x);
				}
			};


			/// Helper for calculating optical flow
			/** @param[in] previousPoints a vector of previous points
				** @param[in] nextPoints a vector next points
				** @param[in] ksize kernel size for filtering out flow vector noise(1, 3, 5, ...). 
				** If ksize < 3; mean value will be used; otherwise, median filter will be used.
				*/
			void CalculateFlowVector(const vector<Point2f>& previousPoints, vector<Point2f>& nextPoints, vector<uchar>& status, vector<Point2f>& flowVector, bool isSmooth = false, int ksize = 1)
			{
				vector<Point2f>().swap (flowVector);
				Point2f vec;
				if (previousPoints.size () > 0)
				{
					flowVector.resize (previousPoints.size());
					// calculate point based optical flow vectors
					transform(nextPoints.begin (), nextPoints.end(), previousPoints.begin (), flowVector.begin (), VectorDiff());
					//for (size_t i = 0; i < previousPoints.size(); ++i)
					//{
					//    vec.x = nextPoints[i].x - previousPoints[i].x;
					//    vec.y = nextPoints[i].y - previousPoints[i].y;
					//    flowVector.push_back (vec);
					//    //mFlowMagnitude.push_back (sqrt(vec.x*vec.x+vec.y*vec.y));
					//}

					// filter out noise
					if (isSmooth)
					{
						if (ksize < 3)  // use mean value for filtering
						{
							vector<float> mag;
							mag.resize (flowVector.size());
							transform (flowVector.begin (), flowVector.end(), mag.begin (), SqrtMagnitude());
							//for (size_t i = 0; i < mFlowVector.size(); ++i)
							//{ 
							//    mag.push_back(sqrt(flowVector[i].x*flowVector[i].x+flowVector[i].y*flowVector[i].y));
							//}
							float magMean = accumulate (mag.begin (), mag.end(), 0.0f);

							magMean /= mag.size ();

							for (size_t i = 0; i < flowVector.size(); ++i)
							{
								if (status[i] == 1 && mag[i] > 2*magMean)
								{
									status[i] = 0;
								}
							}
						}
						else    // use median filtering
						{
							Size fsize = mFrameBuffer[0]->size();
							Mat mag(fsize.height, fsize.width, CV_32FC1, Scalar(0));    /// store flow vector magnitude for filtering
							Mat ang(fsize.height, fsize.width, CV_32FC1, Scalar(0));    /// store flow vector angle for filtering
							Mat magF(fsize.height, fsize.width, CV_32FC1, Scalar(0));
							Mat angF(fsize.height, fsize.width, CV_32FC1, Scalar(0));
							float* pmag = mag.ptr<float>();
							float* pang = ang.ptr<float>();
							for (size_t i = 0; i < previousPoints.size(); ++i)
							{
								//vec.x = nextPoints[i].x - previousPoints[i].x;
								//vec.y = nextPoints[i].y - previousPoints[i].y;
								//mFlowVector.push_back (vec);
								int p = static_cast<int>(previousPoints[i].y)*fsize.width + static_cast<int>(previousPoints[i].x);
								pmag[p] = sqrt(flowVector[i].x*flowVector[i].x+flowVector[i].y*flowVector[i].y);
								pang[p] = atan2(flowVector[i].y, flowVector[i].x);
							}

							// median filtering
							medianBlur (mag, magF, ksize);
							medianBlur (ang, angF, ksize);

							float* pmagF = magF.ptr<float>();
							float* pangF = angF.ptr<float>();

							flowVector.clear();
							// delete outliers from nextPoints
							for (size_t i = 0; i < previousPoints.size(); ++i)
							{
								int p = static_cast<int>(previousPoints[i].y)*fsize.width + static_cast<int>(previousPoints[i].x);

								//float a = cos(pangF[p]);
								//float b = sin(pangF[p]);
								//float c = pmagF[p];

								nextPoints[i].x = previousPoints[i].x + pmagF[p] * cos(pangF[p]);
								nextPoints[i].y = previousPoints[i].y + pmagF[p] * sin(pangF[p]);

								vec.x = nextPoints[i].x - previousPoints[i].x;
								vec.y = nextPoints[i].y - previousPoints[i].y;
								flowVector.push_back (vec);
								//mFlowMagnitude.push_back (sqrt(vec.x*vec.x+vec.y*vec.y));
							}
						}
					}
				}

			}

			long                    mFrameCount;                ///< count the number of frame processed
			int                     mBuffer;                    ///< number of frame delay + 1 between previous and next frame
			Size                    mWindowSize;                ///< size of search window
			deque<Mat*>             mFrameBuffer;               ///< store pointer to the buffered frames
			deque<vector<Point2f> > mMotionPointsBuffer;        ///< store foreground points of the corresponding frame
			vector<Point2f>         mEstimatedPoints;           ///< store estimated points corresponding to mFgPointsBufferFront
			//vector<Point2f>         mPreviousPoints;            ///< store foreground points in the previous frame for flow calculation
			//vector<Point2f>         mEstimatedPoints;           ///< store estimated points of mPreviousPoints
			vector<uchar>           mStatus;              ///< store status to indicate if there is a match between fgpoints points and estimated points     
			vector<Point2f>         mFlowVector;          ///< store corresponding flow vectors
			//vector<float>           mFlowMagnitude;             ///< store magnitudes of flow vectors
		};
        
        
		COpticalFlow::COpticalFlow(const Mat& frame, const vector<Point2f>& points, int buffer, Size winSize /* = cv::Size */)
			:m_impl(new COpticalFlow::Impl())
		{
			m_impl->mFrameCount = 1;
			m_impl->mBuffer = buffer;
			m_impl->mWindowSize = winSize;

			Size fsize = frame.size();
			Mat* pFrameBuffer = new Mat(fsize.height, fsize.width, CV_8UC1);
			frame.copyTo(*pFrameBuffer);

			// assign pointer to deque<Mat*>
			m_impl->mFrameBuffer.push_back (pFrameBuffer);
			// store corresponding foreground points
			m_impl->mMotionPointsBuffer.push_back (points);

		}


		COpticalFlow::~COpticalFlow()
		{
			// dump the buffer
			for (deque<Mat* >::iterator it = m_impl->mFrameBuffer.begin (); it != m_impl->mFrameBuffer.end(); ++it)
			{
				delete *it;
			}
		}
        
		void COpticalFlow::ObtainOpticalFlow(const Mat& nextFrame, vector<Point2f>& nextPoints, int method)
		{
			if (static_cast<int>(m_impl->mFrameBuffer.size ()) < m_impl->mBuffer)
			{
				Size fsize = nextFrame.size();
				Mat* pFrameBuffer = new Mat(fsize.height, fsize.width, CV_8UC1);
				nextFrame.copyTo (*pFrameBuffer);
				m_impl->mFrameBuffer.push_back (pFrameBuffer);
				m_impl->mMotionPointsBuffer.push_back (nextPoints);
				//m_impl->mEstimatedPointsBuffer.push_back (vector<Point2f>());
				//m_impl->mStatusBuffer.push_back (vector<uchar>());
				//m_impl->mFlowVectorBuffer.push_back (vector<Point2f>());
				m_impl->mFrameCount++;
			}
			else
			{
				Size fsize = nextFrame.size();
				// get previous frame pointer for optical flow calculation
				Mat* pFrontFrame = m_impl->mFrameBuffer.front ();
				m_impl->mFrameBuffer.pop_front ();
				nextFrame.copyTo (*pFrontFrame);
				m_impl->mFrameBuffer.push_back (pFrontFrame);
				// get the foreground points in the previous frame
				//m_impl->mPreviousPoints.clear ();
				//m_impl->mPreviousPoints = m_impl->mPointsBuffer.front ();
				//m_impl->mMotionPointsBuffer.pop_front ();
				// number of points need to calculate optical flow

				m_impl->mMotionPointsBuffer.pop_front ();
				m_impl->mMotionPointsBuffer.push_back (nextPoints);

				size_t nPoints = m_impl->mMotionPointsBuffer.front ().size ();
				m_impl->mEstimatedPoints.clear ();
				m_impl->mEstimatedPoints.resize (nPoints);
				m_impl->mStatus.clear ();
				m_impl->mStatus.resize (nPoints);
				vector<float> err(nPoints, 0.0f);
				//m_impl->mEstimatedPoints.clear();
				//m_impl->mEstimatedPoints = m_impl->mPreviousPoints;
				//m_impl->mStatus.clear();

				if (nPoints > 0)
				{
					// calculate pixel based correspondences for the points in prevImg. NB: allow 0 points input
					calcOpticalFlowPyrLK (*m_impl->mFrameBuffer.front (), *m_impl->mFrameBuffer.back (), m_impl->mMotionPointsBuffer.front (), m_impl->mEstimatedPoints, m_impl->mStatus, err, m_impl->mWindowSize, 0);

					// calculate flow vector
					m_impl->CalculateFlowVector (m_impl->mMotionPointsBuffer.front (), m_impl->mEstimatedPoints, m_impl->mStatus, m_impl->mFlowVector, true);
				}

				// store nextFrame and points to the buffer

				//nextFrame.copyTo (*pPreviousFrame);
				//m_impl->mFrameBuffer.push_back (pPreviousFrame);
				//m_impl->mPointsBuffer.push_back (nextPoints);
				m_impl->mFrameCount++;
			}

		}


		void COpticalFlow::GetFlowVectorMagAng (const vector<Point2f>& flowVector, vector<float>& mag, vector<float>& ang)
		{
			mag.resize (flowVector.size());
			ang.resize (flowVector.size());
			transform (flowVector.begin (), flowVector.end (), mag.begin (), Impl::SqrtMagnitude());
			transform (flowVector.begin (), flowVector.end (), ang.begin (), Impl::ATan2());
		}

		void COpticalFlow::DrawArrows (cv::Mat& frame, const std::vector<cv::Point2f>& previousPoints, const std::vector<cv::Point2f>& nextPoints, const std::vector<uchar>& status, float factor /* = 1.0 */, cv::Scalar lineColour /* = cv::Scalar */)
		{
			for (size_t i = 0; i < previousPoints.size (); ++i)
			{
				if (status[i])
				{
					int lineThickness = 1;
					Point p = previousPoints[i];
					Point q = nextPoints[i];

					float ang = atan2(static_cast<float>(q.y) - static_cast<float>(p.y), static_cast<float>(q.x) - static_cast<float>(p.x));
					float mag = sqrt(static_cast<float>(q.y - p.y)*static_cast<float>(q.y - p.y) + static_cast<float>(q.x - p.x)*static_cast<float>(q.x - p.x));

					if (mag < 1.0)
						continue;

					// lengthen the arrow by a factor of one
					q.x = static_cast<int>(p.x + factor*mag*cos(ang));
					q.y = static_cast<int>(p.y + factor*mag*sin(ang));

					// now draw the main line of the arrow
					line(frame, p, q, lineColour, lineThickness);

					// now draw the tips of the arrow
					p.x = static_cast<int>(q.x - factor*cos(ang + CV_PI/4));
					p.y = static_cast<int>(q.y - factor*sin(ang + CV_PI/4));
					line(frame, p, q, lineColour, lineThickness);

					p.x = static_cast<int>(q.x - factor*cos(ang - CV_PI/4));
					p.y = static_cast<int>(q.y - factor*sin(ang - CV_PI/4));
					line(frame, p, q, lineColour, lineThickness);
				}
			}
		}

		bool COpticalFlow::OrientationHistogram(const std::vector<cv::Point2f>& flowVector, const std::vector<uchar>& status, Mat& histogram, bool isNormalised)
		{
			static const float pi = 3.14159f;
			histogram.setTo (0);    // reset histogram
			if (flowVector.size() > 0)
			{
				// histogram is a M x 1 array
				Size hsize = histogram.size();
				float binWidth = 2.0f*pi/hsize.height;     // bin width of the histogram
				int offset = static_cast<int>(ceil(hsize.height/2.0f));
				float mag = 0.0f;       // magnitude of flow
				float ang = 0.0f;        // angle of flow
				float sumMag = 0.0f;
				float* pHist = histogram.ptr<float> ();
				for (size_t i = 0; i < flowVector.size(); ++i)
				{
					if (status[i])
					{
						mag = sqrt(flowVector[i].x*flowVector[i].x + flowVector[i].y*flowVector[i].y);
						ang = atan2(flowVector[i].y, flowVector[i].x);        // in radians -pi to pi

						int index = static_cast<int>(floor(ang/binWidth)) + offset;
						pHist[index] += mag;
						sumMag += mag;
					}
				}

				// to avoid no good match status == all 0
				if (sumMag < 1.0e-6)
					return false;

				if (isNormalised)
				{
					for (int i = 0; i < hsize.height; ++i)
					{
						pHist[i] /= sumMag;
					}
				}


				return true;
			}
			else
			{
				return false;
			}
		}


		const long COpticalFlow::FrameCount() const
		{
			return m_impl->mFrameCount;
		}

		const int COpticalFlow::Buffer() const
		{
			return m_impl->mBuffer;
		}

		const vector<Point2f>& COpticalFlow::MotionPoints () const
		{
			return m_impl->mMotionPointsBuffer.front ();
		}

		const vector<Point2f>& COpticalFlow::EstimatedPoints () const
		{
			return m_impl->mEstimatedPoints;
		}

		const std::vector<uchar>& COpticalFlow::Status() const
		{
			return m_impl->mStatus;
		}

		const std::vector<cv::Point2f>& COpticalFlow::FlowVector() const
		{
			return m_impl->mFlowVector;
		}

		//const std::vector<float>& COpticalFlow::FlowMagnitude() const
		//{
		//    return m_impl->mFlowMagnitude;
		//}

	}
}