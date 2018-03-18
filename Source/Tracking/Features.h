#ifndef TRACKING_FEATURES_H
#define TRACKING_FEATURES_H

#include <opencv2/opencv.hpp>
#include <boost/scoped_ptr.hpp>
#include <vector>


namespace Tracking {
	namespace Features {

		// hold various features of the objects
        struct SObjectFeatures
        {
            // TODO: add other features 
            std::vector<cv::Rect> m_boundingBoxes; // this includes box corners coordinates
			cv::Rect m_latestBoundingBox;	// always equal to the last element in m_boundingBoxes;
            //cv::Mat feature;

            void Release()
            {
                std::vector<cv::Rect>().swap(m_boundingBoxes);
				m_latestBoundingBox = cv::Rect();
                //feature.release();
            }
        };


		/// Optical flow module to extract flow vector or the estimated points in the current frame
		class COpticalFlow
		{
		public:
			/// Ctor of optical flow
			/** 
				** @param[in] frame current frame
				** @param[in] points vector of points to be estimated in the next frame
				** @param[in] buffer buffer-1 = number of frame between previous and next frame used for calculating optical flow
				** @param[in] winSize window size for searching
				**/
			explicit COpticalFlow(const cv::Mat& frame, const std::vector<cv::Point2f>& points, int buffer = 2, cv::Size winSize = cv::Size(21,21));

			/// Dtor of optical flow
			virtual ~COpticalFlow();

			/// Methods for optical flow calculation
			enum {LUCAS_KANADE};

			/// Calculate optical flow
			/** If frame count < delay the function simply push \b nextFrame and \b nextPoints to buffer
				** @param[in] nextFrame next frame for flow vector calculation
				** @param[in] nextPoints corresponding points for flow vector calculation
				** @param[in] method optical flow method
				*/
			void ObtainOpticalFlow(const cv::Mat& nextFrame, std::vector<cv::Point2f>& nextPoints, int method = LUCAS_KANADE);


			void GetFlowVectorMagAng (const std::vector<cv::Point2f>& flowVector, std::vector<float>& mag, std::vector<float>& ang);
			/// Draw arrows
			/** Draw arrows on image to illustrate optical flow vector
				** @param[in] frame image that flow vector will be drawn
				** @param[in] previousPoints a vector of previous positions of observed points
				** @param[in] nextPoints a vector current positions of observed points
				** @param[in] status a vector to indicate if there is a match between corresponding points in the previous and current frame
				** @param[in] factor a scaling factor to control the length of flow vector to be drawn
				** @param[in] lineColour line colour to be drawn
				*/
			static void DrawArrows(cv::Mat& frame, const std::vector<cv::Point2f>& previousPoints, const std::vector<cv::Point2f>& nextPoints, const std::vector<uchar>& status, float factor = 1.0, cv::Scalar lineColour = cv::Scalar(0, 0, 255));

			/// Orientation histogram
			/** Calculate the orientation histogram of flow vector 
				** @param[in] flowVector a vector of optical flow vectors
				** @param[in] status a vector to indicate if the corresponding vector is valid
				** @param[in] isNormalised if true, histogram is normalised; otherwise histogram is unnormalised
				** @param[out] histogram a M x 1 array
				** /return true - histogram constructed, otherwise false
				*/
			static bool OrientationHistogram(const std::vector<cv::Point2f>& flowVector, const std::vector<uchar>& status, cv::Mat& histogram, bool isNormalised = true);

			/// Get the number of frames that has been processed
			const long FrameCount() const;
			/// Get the delay between frames for optical flow calculation
			const int Buffer() const;
			/// Get the points in the previous frame for optical flow calculation
			const std::vector<cv::Point2f>& MotionPoints() const;
			/// Get the corresponding estimated points in the current frame  
			const std::vector<cv::Point2f>& EstimatedPoints() const;
			/// Get the status between previous points and their estimations. If the estimation is good set 1; otherwise 0
			const std::vector<uchar>& Status() const;
			/// Get the flow vector
			const std::vector<cv::Point2f>& FlowVector() const;
			///// Get the magnitude of flow vector
			//const std::vector<float>& FlowMagnitude() const;

		private:
			class Impl; /// Implementation class
			std::unique_ptr<Impl> m_impl;    /// Pointer to implementation
		};
	}
}



#endif
