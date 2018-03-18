///
/// \file
///  \brief Main interface of foreground detection algorithm as described in 
///  http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=5414398&url=http%3A%2F%2Fieeexplore.ieee.org%2Fxpls%2Fabs_all.jsp%3Farnumber%3D5414398
///  \author Hao Li hao.li@live.co.uk
///  \version 1.0
///  \date 2017
///  \pre OpenCV and Boost libraries
///


#ifndef FG_DETECTION_H
#define FG_DETECTION_H

#include "Utils/Export.h"
#include <map>
#include <boost/shared_ptr.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <iosfwd>

//const int GMM_DIMENSION = 3;
//const int GMM_MAX_NUMBER = 5;

#pragma warning(disable: 4251)

namespace FG {

	class APT_API CBgModelBase
	{
	public:
		virtual ~CBgModelBase() {}
		virtual void BgModelUpdate(const cv::Mat& frame, const cv::Mat* regionMask = nullptr) = 0;
		virtual void ShadowHighlightRemoval(const cv::Mat& frame) {};
		virtual void ForegroundNoiseRemoval() {};
		virtual void ForegroundFillHoles() {};
		virtual void BgModelPostUpdate(const cv::Mat& frame) {};
	};


    /// Store initial parameters of GMM based background model
    class APT_API CGmmBgModelParam
    {
    public:
        /// Default ctor
        /** If configuration file is not given, the default parameters will be used
            **/
        CGmmBgModelParam();

		explicit CGmmBgModelParam(const std::string& configFile);

        /// Ctor
        /** @param[in] filename name of the xml config file that stores the background model parameters
            **/ 
        //explicit CGmmBgModelParam(const std::string& filename);

        /// Get the update window size            
        int WindowSize() const;
        /// Get the number of Gaussians used in the model
        int NumberGmm() const;
        /// Get the threshold of standard deviation of Gaussian
        float StdThreshold() const;
        /// Get the background threshold of GMM model
        float BgThreshold() const;
        /// Get the gradient threshold
        float GradientThreshold() const;
        /// Get the shadow brightness threshold for shadow removal
        float ShadowBrightThreshold() const;
        /// Get the highlight threshold for highlight removal
        float HighlightThreshold() const;
        /// Get the global frame update rate for selective region update
        int GlobalUpdateRate() const;
        /// Get the width of minimum region to be removed in foreground map
        int MinRegionWidth() const;
        /// Get the height of minimum region to be removed in foreground map
        int MinRegionHeight() const;
        /// Get the initial sigma square of Gaussian in the background model
        float InitSigma2() const; 
        /// Get the support rate in the background model
        float SupportRate() const;
        /// Get the shadow chroma threshold for shadow removal
        float ShadowChromaThreshold() const;
        /// Get the grid size in the background model
        int GridSize() const;
        /// Get the median filter size for foreground map refinement
        int MedfiltSize() const;
            
        /// Dtor
        virtual ~CGmmBgModelParam() {}

    private:
        class APT_API Impl; /// Implementation class
        boost::shared_ptr<Impl> m_impl;    /// Pointer to implementation
    };

    /// GMM based background model for foreground detection
	class APT_API CGmmBgModel : public CBgModelBase
    {
    public:
        /// Ctor of background model
        /** @param[in] frame frame used to initialise the background model
            ** @param[in] param model initial parameters
            ** @param[in] isCoarseDetection if true, coarse foreground detection will be applied first;
            ** if false, pure pixel based GMM will be used
            **/
        explicit CGmmBgModel(const cv::Mat& frame, const CGmmBgModelParam& param, bool isCoarseDetection);
            
        /// Dtor
        ~CGmmBgModel();
            
        enum {
			GMM_DIMENSION = 3 /**< Dimension of  Gaussian in the model*/, 
            GMM_MAX_NUMBER = 5 /**< Maximum supported number of Gaussians of the model*/
		};
            
        /// Update the corresponding data in the model
        /** @param[in] frame new input frame
            **/
        void BgModelUpdate(const cv::Mat& frame, const cv::Mat* regionMask = nullptr);
            
        /// Remove shadows and highlights in the foreground map
        /** @param[in] frame current frame used for model update
            **/
        void ShadowHighlightRemoval(const cv::Mat& frame);
            
        /// Remove noise in the foreground map
        /** First median filtering the foreground map (GmmBgModelParam::MedfiltSize()), then delete small regions (if < GmmBgModelParam::MinArea())
            **/
        void ForegroundNoiseRemoval();
            
        /// Fill the holes in foreground map
        void ForegroundFillHoles();

        /// Function to wrap GmmBgModel::ShadowHighlightRemoval, GmmBgModel::ForegroundNoiseRemoval and GmmBgModel::ForegroundFillHoles
        /** The function run GmmBgModel::ShadowHighlightRemoval, 
            ** GmmBgModel::ForegroundNoiseRemoval and GmmBgModel::ForegroundFillHoles
            ** in order
            ** @param[in] frame current frame used for model update
            **/
        void BgModelPostUpdate(const cv::Mat& frame);

        /// Find all points in independent foreground regions
        /** @param[in] foreground foreground motion map
            ** @param[out] regions a vector of vectors, each vector in the vector stores the points of an isolated
            ** foreground regions
            ** \return
            **/
        void FindForegroundRegions (const cv::Mat& foreground, std::vector<std::vector<cv::Point> >& regions);

        /// Find bounding boxes of independent foreground regions
        /** @param[in] regions foreground regions
            ** @param[out] boxes a vector of rect, recording the top left corner and box width and height
            ** \return
            **/
        void FindForegroundBoxes (std::vector<std::vector<cv::Point> >& regions, std::vector<cv::Rect>& boxes);

        /// Concatenate foreground points in vector<vector<Point> > regions into a single vector<Point2f>
        /** If there is no foreground points in vector, output vector contains one element Point2f(0.0f, 0.0f)
            ** @param[in] regions a vector of vectors, each vector in the vector stores the points of an isolated
            ** foreground regions
            ** @param[out] points a vector stores all points in the foreground regions
            ** \return
        */
        void GetForegroundPoints(const std::vector<std::vector<cv::Point> >& regions, std::vector<cv::Point2f>& points);
            

		bool IsForgroundOverCrowded() const;

        /// Get the number of frames that has been processed
        const long FrameCount() const;
        /// Get the current background in CV_8UC3
        const cv::Mat& Background() const;
        /// Get the current background in CV_8UC3
        cv::Mat& Background ();
        /// Get the current foreground map in CV_8UC1
        const cv::Mat& Foreground() const;
        /// Get the current foreground map in CV_8UC1
        cv::Mat& Foreground ();
        /// Get the current SgMask in CV_8UC1
        /** If isCoarseDetection == false, return nullptr*/
        const cv::Mat* SgMask() const;
        /// Get the current frame in gray level in CV_8UC1
        /** If isCoarseDetection == false, return nullptr*/
        const cv::Mat* FrameGray() const;
        /// Get the current background in gray level in CV_8UC1
        /** If isCoarseDetection == false, return nullptr*/
        const cv::Mat* BackgroundGray() const;
        /// Get the current frame's gradient in X direction 
        /** If isCoarseDetection == false, return nullptr*/
        const cv::Mat* FrameGradientX() const;
        /// Get the current frame's gradient in Y direction
        /** If isCoarseDetection == false, return nullptr*/
        const cv::Mat* FrameGradientY() const;
        /// Get the magnitude of current frame's gradient
        /** If isCoarseDetection == false, return nullptr*/
        const cv::Mat* FrameGradientMag() const;
        /// Get the current background's gradient in X direction 
        /** If isCoarseDetection == false, return nullptr*/
        const cv::Mat* BackgroundGradientX() const;
        /// Get the current background's gradient in Y direction
        /** If isCoarseDetection == false, return nullptr*/
        const cv::Mat* BackgroundGradientY() const;
        /// Get the magnitude of current background's gradient
        /** If isCoarseDetection == false, return nullptr*/
        const cv::Mat* BackgroundGradientMag() const;
        /// Get the coarse foreground mask in CV_8UC1
        /** If isCoarseDetection == false, return nullptr*/
        const cv::Mat* CFMaskF() const;
        /// Get the update mask to indicate which pixel to be updated in the pixel based detection
        /** If isCoarseDetection == false, return nullptr*/
        const cv::Mat* UpdateMask() const;
        /// Get the points of foreground regions
        /** \return Each vector in the vector stores points of an isolated foreground regions*/
        //const std::vector<std::vector<cv::Point> >& ForegroundRegions() const;
        /// Get the bounding boxes of foreground regions
        /** \return bounding boxes of foreground regions*/
        const std::vector<cv::Rect>& ForegroundBoxes() const;

    private:
        class APT_API Impl; /// Implementation class
        std::unique_ptr<Impl> m_impl;    /// Pointer to implementation
    };

}


#endif