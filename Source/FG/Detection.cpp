///
/// \file
///  \author Hao Li hao.li@live.co.uk 
///  \version 1.0
///  \date 2017
///  \pre OpenCV and Boost libraries
///

#include "Detection.h"
#include <iostream>
#include <vector>
#include <math.h>
#include <deque>
#include <numeric>
#include <algorithm>
#include <functional>
#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>
#include "Utils/Exception.h"
#include "Serialisation/json.h"

using namespace std;
using namespace cv;

namespace {
	class LessRegion : public unary_function<vector<Point>, bool>
    {
    public:
        LessRegion(int minWidth, int minHeight): m_area(minWidth*minHeight) {}
        bool operator()(const vector<Point>& val) const
        {
            return static_cast<int>(val.size()) < m_area;
        }

    private:
        int m_area;
    };
}

namespace FG {
    /// Struct definition of GMM based background model
    class CGmmModelParam
    {
    public:
        // Tuning parameters
        int     m_windowSize;                ///< reciprocal of update rate (300-600)
        int     m_numberGmm;                 ///< number of Gaussians (3-5)
        float   m_stdThreshold;              ///< define in normal Gaussian distribution (mu-std_thre < x < mu+std_thre), m_stdThreshold = 2.5 means 99% included
        float   m_bgThreshold;               ///< threshold of weight sum (0.8-0.9)
        float   m_gradientThreshold;         ///< threshold for pixel gradient similarity measurement (0.7-0.9)
        float   m_shadowBrightThreshold;     ///< threshold for shadow brightness distortion (0.7-0.9)
        float   m_highlightThreshold;        ///< threshold for sudden lighting change (1.1-1.3)
        int     m_globalUpdateRate;          ///< update rate for non-moving regions (5-10)
        int     m_minRegionWidth;            ///< minimum foreground width (5-10) for removing small regions
        int     m_minRegionHeight;           ///< minimum foreground height (10-20)

        // Fixed parameters
        float   m_initSigma2;                ///< Square of standard deviation for each cluster (10)
        float   m_supportRate;               ///< percentage of samples that supports an existed cluster (0.05)
        float   m_shadowChromaThreshold;     ///< threshold for shadow chroma distortion (3.0)
        int     m_gridSize;                  ///< grid size for coarse FG mask generation
        int     m_medfiltSize;               ///< size of 2D median filter
    };
        
    /// CGmmBgModelParam implementation details
    class CGmmBgModelParam::Impl
    {
    public:
        /// helper to set default BgModelParam
        void BgModelParamDefault()
        {
            m_param.m_windowSize = 300;
            m_param.m_numberGmm = 3;
            m_param.m_stdThreshold = 3;
            m_param.m_bgThreshold = 0.90f;
            m_param.m_gradientThreshold = 0.80f;
            m_param.m_shadowBrightThreshold = 0.70f;
            m_param.m_highlightThreshold = 1.20f;
            m_param.m_globalUpdateRate = 10;
            m_param.m_minRegionWidth = 5;
            m_param.m_minRegionHeight = 10;
            m_param.m_initSigma2 = 10;
            m_param.m_supportRate = 0.05f;
            m_param.m_shadowChromaThreshold = 3.0f;
            m_param.m_gridSize = 4;
            m_param.m_medfiltSize = 7;
        }

		void BgModelParamFromConfig(const string& configFile)
		{
			CREQUIRE(!configFile.empty(), "No configuration file specified");

			Json::Reader reader;
			Json::Value root;
			std::ifstream ifs(configFile.c_str());
			CREQUIRE(ifs.is_open(), "Error opening configuration file: " + configFile);
			CREQUIRE(reader.parse(ifs, root), "Error parsing configuration file: " + reader.getFormattedErrorMessages());

			const auto& config = root["ForegroundDetection"];
			const auto& preGmm = config["Preprocessing"];
			const auto& gmm = config["GMM"];
			const auto& postGmm = config["Postprocessing"];
			m_param.m_gradientThreshold = preGmm["GradientThreshold"].asFloat();
			m_param.m_globalUpdateRate = preGmm["GlobalUpdateRate"].asInt();
			m_param.m_gridSize = preGmm["GridSize"].asInt();
			m_param.m_windowSize = gmm["WindowSize"].asInt();
			m_param.m_numberGmm = gmm["NumberGmms"].asInt();
			m_param.m_stdThreshold = gmm["StdThreshold"].asFloat();
			m_param.m_initSigma2 = gmm["InitSigma2"].asFloat();
			m_param.m_bgThreshold = gmm["BackgroundThreshold"].asFloat();
			m_param.m_supportRate = gmm["SupportRate"].asFloat();
			m_param.m_shadowBrightThreshold = postGmm["ShadowBrightThreshold"].asFloat();
			m_param.m_shadowChromaThreshold = postGmm["ShadowChromaThreshold"].asFloat();
			m_param.m_highlightThreshold = postGmm["HighlightThreshold"].asFloat();
			m_param.m_minRegionWidth = postGmm["MinRegionWidth"].asInt();
			m_param.m_minRegionHeight = postGmm["MinRegionHeight"].asInt();
			m_param.m_medfiltSize = postGmm["MedianFilterSize"].asInt();
		}

        CGmmModelParam m_param;
    };


    CGmmBgModelParam::CGmmBgModelParam() : m_impl(new CGmmBgModelParam::Impl())
    {
        m_impl->BgModelParamDefault ();
    }

    CGmmBgModelParam::CGmmBgModelParam (const std::string& configFile) : m_impl(new CGmmBgModelParam::Impl())
    {
		m_impl->BgModelParamFromConfig(configFile);
    }

    // Get all background model parameters
    int     CGmmBgModelParam::WindowSize() const                 {return m_impl->m_param.m_windowSize;}
    int     CGmmBgModelParam::NumberGmm() const                  {return m_impl->m_param.m_numberGmm;}
    float   CGmmBgModelParam::StdThreshold() const               {return m_impl->m_param.m_stdThreshold;}
    float   CGmmBgModelParam::BgThreshold() const                {return m_impl->m_param.m_bgThreshold;}
    float   CGmmBgModelParam::GradientThreshold() const          {return m_impl->m_param.m_gradientThreshold;}
    float   CGmmBgModelParam::ShadowBrightThreshold() const      {return m_impl->m_param.m_shadowBrightThreshold;}
    float   CGmmBgModelParam::HighlightThreshold() const         {return m_impl->m_param.m_highlightThreshold;}
    int     CGmmBgModelParam::GlobalUpdateRate() const           {return m_impl->m_param.m_globalUpdateRate;}
    int     CGmmBgModelParam::MinRegionWidth() const             {return m_impl->m_param.m_minRegionWidth;}
    int     CGmmBgModelParam::MinRegionHeight() const            {return m_impl->m_param.m_minRegionHeight;}
    float   CGmmBgModelParam::InitSigma2() const                 {return m_impl->m_param.m_initSigma2;} 
    float   CGmmBgModelParam::SupportRate() const                {return m_impl->m_param.m_supportRate;}
    float   CGmmBgModelParam::ShadowChromaThreshold() const      {return m_impl->m_param.m_shadowChromaThreshold;}
    int     CGmmBgModelParam::GridSize() const                   {return m_impl->m_param.m_gridSize;}
    int     CGmmBgModelParam::MedfiltSize() const                {return m_impl->m_param.m_medfiltSize;}


    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


     
    /// CGmmBgModel implementation details
    class CGmmBgModel::Impl
    {
    public:

        /// Store GMM information at each pixel position
        class CGmmInfo
        {
        public:
            long m_matchCout;				///< number of match times, used before window size reaches predefined value
            float m_updateRate;		///< update rate of current pixel location, can be changed through process
            float m_weight;					///<weight of each Gaussian cluster
            float m_mu[CGmmBgModel::GMM_DIMENSION];		///<store mu of each color channel
            float m_sigma2[CGmmBgModel::GMM_DIMENSION];	///<store Sigma of each color channel
        };

        /// Store GMM information of the model 
        class CGmmPixelInfo
        {
        public:
            int M;      ///< number of Gaussian used in current pixel position
            CGmmInfo* m_info;
        };

        /// Hold grid based coarse foreground detection results
        class CGmmTempInfo
        {
        public:
            shared_ptr<Mat> m_frameGray;
			shared_ptr<Mat> m_backgroundGray;
			shared_ptr<Mat> m_frameGradientX;
			shared_ptr<Mat> m_frameGradientY;
			shared_ptr<Mat> m_backgroundGradientX;
			shared_ptr<Mat> m_backgroundGradientY;
			shared_ptr<Mat> m_frameGradientMag;
			shared_ptr<Mat> m_frameGradientMag2;
			shared_ptr<Mat> m_backgroundGradientMag;
			shared_ptr<Mat> m_backgroundGradientMag2;
			shared_ptr<Mat> m_gradientDotProduct;
			shared_ptr<Mat> m_temp1;
			shared_ptr<Mat> m_temp2;

        };

        /// Calculate similarity gradient mask
        void SgMaskCalculation(const Mat& frame)
        {
            int gradSize = m_param.GridSize();
            int channels = frame.channels();
            if (channels == 3)  // covert to gray scale
            {
                cvtColor (frame, *mpTempInfo->m_frameGray, CV_BGR2GRAY);
                cvtColor (*mpBackground, *mpTempInfo->m_backgroundGray, CV_BGR2GRAY);
            }
            else    // store to temporal place
            {
                frame.copyTo(*mpTempInfo->m_frameGray);
                mpBackground->copyTo(*mpTempInfo->m_backgroundGray);
            }
                
            // calculate the image derivatives using sobel filter
            // TODO gpu::GpuMat to accrelate speed
            Sobel (*mpTempInfo->m_frameGray, *mpTempInfo->m_frameGradientX, CV_32F, 1, 0, 3, 1.0/8);    //scaling
            Sobel (*mpTempInfo->m_frameGray, *mpTempInfo->m_frameGradientY, CV_32F, 0, 1, 3, 1.0/8);
            Sobel (*mpTempInfo->m_backgroundGray, *mpTempInfo->m_backgroundGradientX, CV_32F, 1, 0, 3, 1.0/8);
            Sobel (*mpTempInfo->m_backgroundGray, *mpTempInfo->m_backgroundGradientY, CV_32F, 0, 1, 3, 1.0/8);

            // find the square of magnitude of gradient
            multiply(*mpTempInfo->m_frameGradientX, *mpTempInfo->m_frameGradientX, *mpTempInfo->m_temp1);
            multiply(*mpTempInfo->m_frameGradientY, *mpTempInfo->m_frameGradientY, *mpTempInfo->m_temp2);
            add (*mpTempInfo->m_temp1, *mpTempInfo->m_temp2, *mpTempInfo->m_frameGradientMag2);
            multiply(*mpTempInfo->m_backgroundGradientX, *mpTempInfo->m_backgroundGradientX, *mpTempInfo->m_temp1);
            multiply(*mpTempInfo->m_backgroundGradientY, *mpTempInfo->m_backgroundGradientY, *mpTempInfo->m_temp2);
            add (*mpTempInfo->m_temp1, *mpTempInfo->m_temp2, *mpTempInfo->m_backgroundGradientMag2);

            Size fsize = frame.size();
            // calculate the dot product and magnitude of gradient;
            float* pGradientDotProduct = mpTempInfo->m_gradientDotProduct->ptr<float>();
            float* pBackgroundGradientX =  mpTempInfo->m_backgroundGradientX->ptr<float>();
            float* pBackgroundGradientY = mpTempInfo->m_backgroundGradientY->ptr<float>();
            float* pFrameGradientMag = mpTempInfo->m_frameGradientMag->ptr<float>();
            float* pFrameGradientMag2 = mpTempInfo->m_frameGradientMag2->ptr<float>();
            float* pBackgroundGradientMag = mpTempInfo->m_backgroundGradientMag->ptr<float>();
            float* pBackgroundGradientMag2 = mpTempInfo->m_backgroundGradientMag2->ptr<float>();
            for (int i = 0, n = 0; i < fsize.height; ++i)
            {
                for (int j = 0; j < fsize.width; ++j, ++n)
                {
                    // dot product
                    pGradientDotProduct[n] = pBackgroundGradientX[n]*pBackgroundGradientX[n] + pBackgroundGradientY[n]*pBackgroundGradientY[n];
                    // gradient
                    pFrameGradientMag[n] = sqrt(pFrameGradientMag2[n]);
                    pBackgroundGradientMag[n] = sqrt(pBackgroundGradientMag2[n]);
                }
            }

            //Scalar_<float> frameGradientMean = mean(*mpTempInfo->m_frameGradientMag);
            //Scalar_<float> backgroundGradientMean = mean(*mpTempInfo->m_backgroundGradientMag);
            //cout << "\"" << frameGradientMean(0) << " " << backgroundGradientMean(0) << "\"";
            // calculate similarity mask SgMask == 1 for gradient dissimilar regions, 0 for similar regions and gradient homogeneous regions
            int gridSize = m_param.GridSize ();
            //int nElems = gridSize*gridSize;
            uchar* pSgMask = mpSgMask->ptr<uchar>();
            for (int i = 0, n = 0; i < fsize.height/gridSize; ++i)
            {
                for (int j = 0; j < fsize.width/gridSize; ++j, ++n)
                {
                    float sum1 = 0.0f;
                    float sum2 = 0.0f;
                    //float sum3 = 0.0f;  /// mean of magnitude of each grid in frame
                    //float sum4 = 0.0f;  /// mean of magnitude of each grid in background
                    for (int x = 0; x < gridSize; ++x)
                    {
                        for (int y = 0; y < gridSize; ++y)
                        {
                            int index = (i*gridSize+x)*fsize.width+j*gridSize+y;
                            //sum1 += 2*mpTempInfo->m_gradientDotProduct->at<float>(i*gridSize+x, j*gridSize+y);
                            //sum2 += mpTempInfo->m_frameGradientMag2->at<float>(i*gridSize+x, j*gridSize+y) + mpTempInfo->m_backgroundGradientMag2->at<float>(i*gridSize+x, j*gridSize+y);
                            sum1 += 2*pGradientDotProduct[index];
                            sum2 += pFrameGradientMag2[index]+pBackgroundGradientMag2[index];
                            //sum3 += pFrameGradientMag[n];
                            //sum4 += pBackgroundGradientMag[n];
                        }
                    }

                    //sum3 /= nElems;
                    //sum4 /= nElems;

                    //float thres = 0.5f; /// to remove homogeneous regions
                    if (/*(sum3 > thres || sum4 > thres) && */sum1/sum2 < m_param.GradientThreshold ())
                    {
                        pSgMask[n] = static_cast<uchar>(255);
                    } else
                    {
                        pSgMask[n] = static_cast<uchar>(0);	//TODO
                    }

                }
            }

        }

        /// Coarse foreground detection based on single Gaussian model
        void CoarseForegroundDetection(const Mat& frame)
        {
            Size fsize = frame.size();
            int channels = frame.channels ();
            int gridSize = m_param.GridSize ();
            int nElems = gridSize*gridSize;
            float stdThres = m_param.StdThreshold ();
            const uchar* pdata = frame.ptr<uchar>();
            uchar* pCFMaskS = mpCFMaskS->ptr<uchar>();
            float* pCBMu = mpCBMu->ptr<float>();
            float* pCBSigma2 = mpCBSigma2->ptr<float>();
            float* pCFMu = mpCFMu->ptr<float>(); 
            // Gaussian based detection
            for (int i = 0, n = 0; i < fsize.height/gridSize; ++i)
            {
                for (int j = 0; j < fsize.width/gridSize; ++j, ++n)
                {
                    float sumCBMu[CGmmBgModel::GMM_DIMENSION] = {0.0f};
                    float sumCBSigma2[CGmmBgModel::GMM_DIMENSION] = {0.0f};
                    float sumCFMu[CGmmBgModel::GMM_DIMENSION] = {0.0f};

                    // sum in each grid, take the 1st Gaussian in the GMM as the current background
                    for (int x = 0; x < gridSize; ++x)
                    {
                        for (int y = 0; y < gridSize; ++y)
                        {
                            int index = (i*gridSize+x)*fsize.width + j*gridSize+y;
                            int p = index*channels;

                            for (int d = 0; d < channels; ++d)
                            {
                                sumCBMu[d] += mpPixelInfo[index].m_info[0].m_mu[d];
                                sumCBSigma2[d] += mpPixelInfo[index].m_info[0].m_sigma2[d];
                                sumCFMu[d] += static_cast<float>(pdata[p+d]);
                            }
                        }
                    }

                    float sumDelta2 = 0.0f;
                    float sum_sigma2 = 0.0f;
                    // average each grid and do single Gaussian detection
                    for (int d = 0; d < channels; ++d)
                    {
                        pCBMu[n+d] = sumCBMu[d]/nElems;
                        pCBSigma2[n+d] = sumCBSigma2[d]/nElems;
                        pCFMu[n+d] = sumCFMu[d]/nElems;

                        float delta = sumCFMu[d] - sumCBMu[d];
                        sumDelta2 += delta*delta;
                        sum_sigma2 += sumCBSigma2[d];
                    }
                    sum_sigma2 = stdThres*stdThres*sum_sigma2;

                    // generate coarse foreground mask
                    if (sumDelta2 >= sum_sigma2)
                    {
                        pCFMaskS[n] = static_cast<uchar>(255);
                    } else
                    {
                        pCFMaskS[n] = static_cast<uchar>(0);
                    }
                }
            }

            // extend CFMaskS to full size CFMaskF and generate a updateMask based on CFMaskF and SgMask
            Size sMaskSize = mpCFMaskS->size();
            uchar* pCFMaskF = mpCFMaskF->ptr<uchar>();
            uchar* pUpdateMask = mpUpdateMask->ptr<uchar>();
            uchar* pSgMask = mpSgMask->ptr<uchar>();
            for (int i = 0, n = 0; i < fsize.height; ++i)
            {
                for (int j = 0; j < fsize.width; ++j, ++n)
                {
                    int index = i/gridSize*sMaskSize.width + j/gridSize;
                    pCFMaskF[n] = pCFMaskS[index];
                    pUpdateMask[n] = pSgMask[index];
                }
            }
                
            Mat key = Mat::ones(3, 3, CV_8U);

            dilate(*mpCFMaskF, *mpCFMaskF, key, Point(-1, -1), 1);
            bitwise_or(*mpCFMaskF, *mpUpdateMask, *mpUpdateMask);


        }

        /// Calculate probability of pixel follows a Gaussian distribution
        /** @param[in] pixel pixel value in a \b d dimensional vector
            ** @param[in] pInfo reference to CGmmInfo
            ** @param[in] channels number of dimensions of the pixel
            ** \return Gaussian probability
            */
        float GaussianPDF(const float* pixel, const CGmmInfo& pInfo, int channels = 3)
        {
            static const float pi = 3.14159f;
            float invSigma2 = 0.0f;
            float invDetSigma2 = 1.0;
            float delta = 0.0f;
            float sumVec = 0.0f;

            for (int d = 0; d < channels; ++d)
            {
                delta = pixel[d] - pInfo.m_mu[d];
                invSigma2 = 1.0f/pInfo.m_sigma2[d];
                invDetSigma2 *= invSigma2;
                sumVec += delta*delta*invSigma2;
            }

            return static_cast<float>(sqrt(pow(2.0*pi, -channels)*invDetSigma2)*exp(-0.5*sumVec));
        }

        /// Test if current pixel matches any Gaussian in the model
        /** @param[in] pixel pixel value
            ** @param[in] pPixelInfo pointer to CGmmPixelInfo
            ** @param[in] channels number of dimensions of the pixel
            ** @param[out] matchIndex a <b>1 x M</b> array (\b M is the number of Gaussians in GMM). The 
            ** corresponding match position is set to 1, otherwise all 0
            ** \return \b true if has match, otherwise \b false
            ***/
        bool GmmMatchTest(const float* pixel, const CGmmPixelInfo* pPixelInfo, int* matchIndex, int channels = 3)
        {
            bool isMatch = false;
            const int M = pPixelInfo->M;
            float stdThres = m_param.StdThreshold ();
            // each Gaussian is already sorted in descending order
            for (int k = 0; k < M; ++k)
            {
                float sumDelta2 = 0.0f;
                float sum_sigma2 = 0.0f;
                for (int d = 0; d < channels; ++d)
                {
                    float delta = pixel[d] - pPixelInfo->m_info[k].m_mu[d];
                    sumDelta2 += delta*delta;
                    sum_sigma2 += pPixelInfo->m_info[k].m_sigma2[d];
                }
                sum_sigma2 = stdThres*stdThres*sum_sigma2;

                // match measure under Euclidean distance and extract the number of first matched Gaussian
                if (sumDelta2 < sum_sigma2)
                {
                    matchIndex [k] = 1;
                    isMatch = true;
                    break;
                }
            }

            return isMatch;

        }

        /// Update the GMM model at each pixel position if has matched input
        void Gm_mupdateMatch(const float* pixel, CGmmPixelInfo* pPixelInfo, const int* matchIndex, bool isGlobalUpdate, uchar updateMask, long frameCount, int channels = 3)
        {
            
            int windowSize = m_param.WindowSize();
            float supportRate = m_param.SupportRate ();
            float learningRate;
            if (isGlobalUpdate && !updateMask)
            {
                learningRate = frameCount < windowSize ? 1.0f/frameCount : m_param.GlobalUpdateRate()*1.0f/windowSize;
            }
            else
            {
                learningRate = frameCount < windowSize ? 1.0f/frameCount : 1.0f/windowSize;
            }
            float weightSum = 0.0f;
            float delta = 0.0f;
            const int M = pPixelInfo->M;
            float updateRate = 0.0f;
            float probGaussian = 0.0f;
            float eat = 0.0f;

            for (int k = 0; k < M; ++k)
            {
                pPixelInfo->m_info[k].m_matchCout += static_cast<long>(matchIndex[k]);

                // gmm initialisation stage
                if (frameCount < windowSize)    
                {
                    pPixelInfo->m_info[k].m_weight += learningRate*(static_cast<float>(matchIndex[k]) - pPixelInfo->m_info[k].m_weight);
                } 
                else  // gmm enters stable stage
                {
                    pPixelInfo->m_info[k].m_weight += learningRate*(static_cast<float>(matchIndex[k]) - pPixelInfo->m_info[k].m_weight) - learningRate*supportRate;
                    // delete negative weighted clusters
                    if ( pPixelInfo->m_info[k].m_weight < 0)
                    {
                        pPixelInfo->M--;
                        pPixelInfo->m_info[k].m_matchCout = 0;
                        pPixelInfo->m_info[k].m_weight = 0;

                        for (int d = 0; d < channels; ++d)
                        {
                            pPixelInfo->m_info[k].m_mu[d] = 0;
                            pPixelInfo->m_info[k].m_sigma2[d] = 0;
                        }
                    }
                }

                weightSum += pPixelInfo->m_info[k].m_weight;
                    
                if (matchIndex[k])
                {
                    updateRate = frameCount < windowSize ? static_cast<float>(matchIndex[k])/static_cast<float>(pPixelInfo->m_info[k].m_matchCout) : learningRate*GaussianPDF (pixel, pPixelInfo->m_info[k], channels);

                        
                    for (int d = 0; d < channels; ++d)
                    {
                        delta = pixel[d] - pPixelInfo->m_info[k].m_mu[d];
                        pPixelInfo->m_info[k].m_mu[d] += updateRate*delta;
                        pPixelInfo->m_info[k].m_sigma2[d] += updateRate*(delta*delta - pPixelInfo->m_info[k].m_sigma2[d]);

                        // avoid divergence? TODO?
                    }
                }
            }

            // normalise the weight of each Gaussian
            for (int k = 0; k < M; ++k)
            {
                pPixelInfo->m_info[k].m_weight /= weightSum;
            }

        }

        /// Update GMM information at each pixel position if no matched input found
        void Gm_mupdateNoMatch(const float* pixel, CGmmPixelInfo* pPixelInfo, long frameCount, int channels = 3)
        {
            int windowSize = m_param.WindowSize();
            float learningRate = frameCount < windowSize ? 1.0f/frameCount : 1.0f/windowSize;
            int M = pPixelInfo->M;
            float initSigma2 = m_param.InitSigma2 ();
            float weightTemp = 0.0f;
                
            if (M < m_param.NumberGmm ())
            {
                pPixelInfo->M++;
                M++;
            }
            else
            {
                weightTemp = pPixelInfo->m_info[M-1].m_weight;
            }

            pPixelInfo->m_info[M-1].m_matchCout = 1;
            pPixelInfo->m_info[M-1].m_weight = learningRate;

            for (int d = 0; d < channels; ++d)
            {
                pPixelInfo->m_info[M-1].m_mu[d] = pixel[d];
                pPixelInfo->m_info[M-1].m_sigma2[d] = initSigma2;
            }

            for (int k = 0; k < M; ++k)
            {
                pPixelInfo->m_info[k].m_weight /= 1.0f - weightTemp + learningRate;
            }

        }

        /// Sort the GMM at each pixel position in descending order according to each Gaussian's weight/Sigma2
        void GmmSort(CGmmPixelInfo* pPixelInfo, int* matchIndex, int channels = 3)
        {
            const int M = pPixelInfo->M;
            float sortIndex[CGmmBgModel::GMM_MAX_NUMBER] = {0.0f};
            for (int k = 0; k < M; ++k)
            {
                // avoid divided by zero in initial step and after delete some clusters
                if (pPixelInfo->m_info[k].m_matchCout > 0)
                {
                    float sum_sigma2 = 0.0f;
                    for (int d = 0; d < channels; ++d)
                    {
                        sum_sigma2 += pPixelInfo->m_info[k].m_sigma2[d];
                    }

                    sortIndex[k] = pPixelInfo->m_info[k].m_weight/sum_sigma2;
                }

            }

            // reorder Gaussians according to the index
            for (int k = 1; k < M; ++k)
            {
                for (int i = k; i > 0 && (sortIndex[i-1] < sortIndex[k]); --i)
                {
                    float sortIndexTemp = sortIndex[i];
                    sortIndex[i] = sortIndex[i-1];
                    sortIndex[i-1] = sortIndexTemp;

                    int matchIndexTemp = matchIndex[i];
                    matchIndex[i] = matchIndex[i-1];
                    matchIndex[i-1] = matchIndexTemp;

                    // using pointer change to speed up?
                    CGmmInfo infoTemp = pPixelInfo->m_info[i];
                    pPixelInfo->m_info[i] = pPixelInfo->m_info[i-1];
                    pPixelInfo->m_info[i-1] = infoTemp;
                }
            }

        }

        /// Generate current background image from GMM model
        void GmmBackground()
        {
            Size fsize = mpBackground->size();
            int channels = mpBackground->channels ();
            uchar* pBackground = mpBackground->ptr<uchar>();
            for (int i = 0, n = 0; i < fsize.height; ++i)
            {
                for (int j = 0; j < fsize.width; ++j, ++n)
                {
                    int p = n*channels;
                    for (int d = 0; d < channels; ++d)
                    {
                        pBackground[p+d] = static_cast<uchar>(mpPixelInfo[n].m_info[0].m_mu[d]+0.5);
                    }
                }
            }
        }

        /// Calculate if current pixel position belongs to foreground and set the corresponding mask in mpForeground
        void GmmForegourndPixel(const CGmmPixelInfo* pPixelInfo, const int* matchIndex, int n, int regionMask)
        {
            uchar foregroundPixel = static_cast<uchar>(255);
            float weightSum = 0.0f;
            int M = pPixelInfo->M;
            float bgThres = m_param.BgThreshold ();
            for (int k = 0; k < M; ++k)
            {
                if (matchIndex[k])
                {
                    foregroundPixel = 0;
                }

                weightSum += pPixelInfo->m_info[k].m_weight;
                if (weightSum > bgThres)
                    break;
            }

            if (mIsCoarseDetection)
                mpForeground->at<uchar>(n) = foregroundPixel & mpCFMaskF->at<uchar>(n) & regionMask;
            else
                mpForeground->at<uchar>(n) = foregroundPixel & regionMask;

        }

        /// Find connected regions from foreground
        void Neighbours(const cv::Mat& foreground, std::vector<cv::Point>& points, int previndex, int index, int nNeighbours = 8)
        {
            Size fsize = foreground.size();
            const uchar* pdata = foreground.ptr<uchar>();
            uchar* ptemp = mpTempMat->ptr<uchar>(); // to record if the current foreground pixel has been looped

            if (index < 0 || index >= fsize.height*fsize.width)
            {
                return;
            }
            else
            {
                if (pdata[index] > 0 && ptemp[index] == 0)  // good to go
                {
                    if (index%fsize.width == 0 && previndex%fsize.width == fsize.width-1) // central pixel at right boundary
                        return;
                    if (index%fsize.width == fsize.width-1 && previndex%fsize.width == 0) // left boundary
                        return;

                    ptemp[index] = 1;
                    Point point;
                    point.x = index%fsize.width;
                    point.y = index/fsize.width;
                    points.push_back (point);

                    if (nNeighbours == 4)
                    {
                        Neighbours (foreground, points, index, index-1, 4);
                        Neighbours (foreground, points, index, index+1, 4);
                        Neighbours (foreground, points, index, index-fsize.width, 4);
                        Neighbours (foreground, points, index, index+fsize.width, 4);
                    }
                    else if (nNeighbours == 8)
                    {
                        Neighbours (foreground, points, index, index-1, 8);
                        Neighbours (foreground, points, index, index+1, 8);
                        Neighbours (foreground, points, index, index-fsize.width, 8);
                        Neighbours (foreground, points, index, index+fsize.width, 8);
                        Neighbours (foreground, points, index, index-fsize.width-1, 8);
                        Neighbours (foreground, points, index, index-fsize.width+1, 8);
                        Neighbours (foreground, points, index, index+fsize.width-1, 8);
                        Neighbours (foreground, points, index, index+fsize.width+1, 8);
                    }
                }
                return;
            }
        }

        CGmmBgModelParam        m_param;                 ///< store parameters
        long                    mFrameCount;            ///< number of frames that has been processed
        bool                    mIsCoarseDetection;     ///< if ture, do coarse pre-detection, otherwise do simple GMM
        CGmmPixelInfo*          mpPixelInfo;            ///< pointer to array of CGmmPixelInfo (equal to frame size)
        shared_ptr<Mat>                    mpBackground;           ///< store background of the model (channel = channel of frame)
        shared_ptr<Mat>                    mpForeground;           ///< detected foreground (single channel)
        shared_ptr<Mat>                    mpUpdateMask;           ///< mask to indicate which pixel to be updated
        shared_ptr<Mat>                    mpSgMask;               ///< similar gradient mask, smaller sized image, depend on param.gridsize
        shared_ptr<Mat>                    mpCFMaskS;              ///< coarse foreground mask smaller sized
        shared_ptr<Mat>                    mpCFMaskF;              ///< coarse foreground mask full sized
        shared_ptr<Mat>                    mpCBMu;                 ///< coarse background mu
        shared_ptr<Mat>                    mpCBSigma2;             ///< coarse background Sigma2
        shared_ptr<Mat>                    mpCFMu;                 ///< coarse frame mu
        shared_ptr<Mat>                    mpTempMat;              ///< same size as frame for any temp useage
        shared_ptr<CGmmTempInfo>           mpTempInfo;             ///< store intermediate results
        vector<vector<Point> >  mFgContours;            ///< store each individual foreground contours, default initialisation equals to 0
        vector<vector<Point> >  mFgRegions;               ///< store all foreground pixels in each individual foreground regions
        vector<Rect>            mFgBoxes;                 ///< foreground bounding boxes
    };


    CGmmBgModel::CGmmBgModel(const cv::Mat& frame, const CGmmBgModelParam& param, bool isCoarseDetection)
        : m_impl(new CGmmBgModel::Impl())
    {
        // allocate corresponding memories for foreground detection
        int channels = frame.channels (); // number of channel of frame
        Size fsize = frame.size();    // height and width of frame

        m_impl->m_param = param;
        m_impl->mFrameCount = 0;
        m_impl->mIsCoarseDetection = isCoarseDetection;
        m_impl->mpPixelInfo = new Impl::CGmmPixelInfo[fsize.height*fsize.width];
        //const int sizeND1[] = {fsize.height, fsize.width, channels};
        //m_impl->mpBackground = new Mat(channels, sizeND1, CV_8U);
        //frame.copyTo(*m_impl->mpBackground);
        if (1 == channels)
        {
            m_impl->mpBackground.reset(new Mat(fsize.height, fsize.width, CV_8UC1));
        }
        else if (3 == channels)
        {
            m_impl->mpBackground.reset(new Mat(fsize.height, fsize.width, CV_8UC3));
        }

        m_impl->mpForeground.reset(new Mat(fsize.height, fsize.width, CV_8UC1));
        m_impl->mpTempMat.reset(new Mat(fsize.height, fsize.width, CV_8UC1));
        m_impl->mpUpdateMask.reset(new Mat(fsize.height, fsize.width, CV_8UC1));

        if (isCoarseDetection)   // do my method, otherwise pure GMM
        {
            m_impl->mpSgMask.reset(new Mat(fsize.height/param.GridSize (), fsize.width/param.GridSize (), CV_8UC1));
            m_impl->mpCFMaskS.reset(new Mat(fsize.height/param.GridSize (), fsize.width/param.GridSize (), CV_8UC1));
            m_impl->mpCFMaskF.reset(new Mat(fsize.height, fsize.width, CV_8UC1));
            if (1 == channels)
            {                
                m_impl->mpCBMu.reset(new Mat(fsize.height/param.GridSize (), fsize.width/param.GridSize (), CV_32FC1));
                m_impl->mpCBSigma2.reset(new Mat(fsize.height/param.GridSize (), fsize.width/param.GridSize (), CV_32FC1));
                m_impl->mpCFMu.reset(new Mat(fsize.height/param.GridSize (), fsize.width/param.GridSize (), CV_32FC1));
            } else if (3 == channels)
            {
                m_impl->mpCBMu.reset(new Mat(fsize.height/param.GridSize (), fsize.width/param.GridSize (), CV_32FC3));
                m_impl->mpCBSigma2.reset(new Mat(fsize.height/param.GridSize (), fsize.width/param.GridSize (), CV_32FC3));
                m_impl->mpCFMu.reset(new Mat(fsize.height/param.GridSize (), fsize.width/param.GridSize (), CV_32FC3));
            }

        }


        // allocate memories for CGmmInfo class in each pixel position
        //for (int i = 0; i < fsize.height; ++i)
        //{
        //    for (int j = 0; j < fsize.width; ++j)
        //    {
        //        m_impl->mspPixelInfo[i*fsize.width+j].m_info = new Impl::CGmmInfo[param.NumberGmm()];
        //    }
        //}
            
        // assign memories for CGmmInfo as a chunk to reduce assign and delete time
        m_impl->mpPixelInfo[0].m_info = new Impl::CGmmInfo[param.NumberGmm ()*fsize.height*fsize.width];


        const uchar* pdata = frame.ptr();   // pointer to image
        uchar* pbackground = m_impl->mpBackground->ptr ();
        // assign initial value to the model. Foreground, UpdateMask, SgMask, CFMask can be kept as default
        for (int i = 0, n=0; i < fsize.height; ++i)
        {
            for (int j = 0; j < fsize.width; ++j, ++n)
            {
                int p = n*channels;
                m_impl->mpPixelInfo[n].M = 1;
                // pointer offset assigned to m_info in each mpPixelInfo
                m_impl->mpPixelInfo[n].m_info = m_impl->mpPixelInfo[0].m_info + n*param.NumberGmm ();
                m_impl->mpPixelInfo[n].m_info[0].m_matchCout = 1;
                m_impl->mpPixelInfo[n].m_info[0].m_weight = 1;
                m_impl->mpPixelInfo[n].m_info[0].m_updateRate = 1.0f/param.WindowSize();
                for (int d = 0; d < channels; ++d)
                {
                    m_impl->mpPixelInfo[n].m_info[0].m_mu[d] = pdata[p+d];
                    m_impl->mpPixelInfo[n].m_info[0].m_sigma2[d] = param.InitSigma2();
                    pbackground[p+d] = pdata[p+d];
                }


                for (int m = 1; m < param.NumberGmm(); ++m)
                {
                    m_impl->mpPixelInfo[n].m_info[m].m_weight = 0;
                    m_impl->mpPixelInfo[n].m_info[m].m_matchCout = 0;
                    for (int d = 0; d < channels; ++d)
                    {
                        m_impl->mpPixelInfo[n].m_info[m].m_mu[d] = 0.0f;
                        m_impl->mpPixelInfo[n].m_info[m].m_sigma2[d] = 0.0f;
                    }
                }
            }
        }

        if (isCoarseDetection)
        {
            
            int gridSzie = param.GridSize();
            int nElems = gridSzie*gridSzie;
            float* pCBMu = m_impl->mpCBMu->ptr<float>();
            float* pCBSigma2 = m_impl->mpCBSigma2->ptr<float>();
            float* pCFMu = m_impl->mpCFMu->ptr<float>();
            for (int i = 0, n = 0; i < fsize.height/gridSzie; ++i)
            {
                for (int j = 0; j < fsize.width/gridSzie; ++j, ++n)
                {
                    int p = n*channels;
                    float sumCBMu[CGmmBgModel::GMM_DIMENSION] = {0.0f};
                    float sumCBSigma2[CGmmBgModel::GMM_DIMENSION] = {0.0f};
                    float sumCFMu[CGmmBgModel::GMM_DIMENSION] = {0.0f};

                    for (int x = 0; x < gridSzie; ++x)
                    {
                        for (int y = 0; y < gridSzie; ++y)
                        {
                            int n = (i*gridSzie+x)*fsize.width+(j*gridSzie+y);
                            int p = n*channels;
                            for (int d = 0; d < channels; ++d)
                            {
                                sumCBMu[d] +=  m_impl->mpPixelInfo[n].m_info[0].m_mu[d];
                                sumCBSigma2[d] +=  m_impl->mpPixelInfo[n].m_info[0].m_sigma2[d];
                                sumCFMu[d] += pdata[p+d];
                            }
                        }
                    }

                    // average value within each grid for coarse single Gaussian detection
                    for (int d = 0; d < channels; ++d)
                    {
                        sumCBMu[d] /= nElems;
                        sumCBSigma2[d] /= nElems;
                        sumCFMu[d] /= nElems;

                        pCBMu[p+d] = sumCBMu[d];
                        pCBSigma2[p+d] = sumCBSigma2[d];
                        pCFMu[p+d] = sumCFMu[d];

                    }
                }
            }
        }

        if (isCoarseDetection)
        {
            m_impl->mpTempInfo.reset(new Impl::CGmmTempInfo());
			m_impl->mpTempInfo->m_frameGray.reset(new Mat(fsize.height, fsize.width, CV_8UC1));
            m_impl->mpTempInfo->m_backgroundGray.reset(new Mat(fsize.height, fsize.width, CV_8UC1));
            m_impl->mpTempInfo->m_frameGradientX.reset(new Mat(fsize.height, fsize.width, CV_32FC1));
            m_impl->mpTempInfo->m_frameGradientY.reset(new Mat(fsize.height, fsize.width, CV_32FC1));
            m_impl->mpTempInfo->m_backgroundGradientX.reset(new Mat(fsize.height, fsize.width, CV_32FC1));
            m_impl->mpTempInfo->m_backgroundGradientY.reset(new Mat(fsize.height, fsize.width, CV_32FC1));
            m_impl->mpTempInfo->m_frameGradientMag.reset(new Mat(fsize.height, fsize.width, CV_32FC1));
            m_impl->mpTempInfo->m_frameGradientMag2.reset(new Mat(fsize.height, fsize.width, CV_32FC1));
            m_impl->mpTempInfo->m_backgroundGradientMag.reset(new Mat(fsize.height, fsize.width, CV_32FC1));
            m_impl->mpTempInfo->m_backgroundGradientMag2.reset(new Mat(fsize.height, fsize.width, CV_32FC1));
            m_impl->mpTempInfo->m_gradientDotProduct.reset(new Mat(fsize.height, fsize.width, CV_32FC1));
            m_impl->mpTempInfo->m_temp1.reset(new Mat(fsize.height, fsize.width, CV_32FC1));
            m_impl->mpTempInfo->m_temp2.reset(new Mat(fsize.height, fsize.width, CV_32FC1));
        }


        m_impl->mFrameCount++;
        //cout << "CGmmBgModel constructor called" << endl;
    }
    
    CGmmBgModel::~CGmmBgModel()
    {

        //Size fsize = m_impl->mpBackground->size();
        //for (int i = 0; i < fsize.height; ++i)
        //{
        //    for (int j = 0; j < fsize.width; ++j)
        //    {
        //        delete[] m_impl->mpPixelInfo[i*fsize.width+j].m_info;
        //    }
        //}
        delete[] m_impl->mpPixelInfo[0].m_info;
        delete[] m_impl->mpPixelInfo;

    }

    void CGmmBgModel::BgModelUpdate (const cv::Mat& frame, const cv::Mat* regionMask)
    {
        m_impl->mpForeground->setTo(0);   // reset foreground
        m_impl->mpTempMat->setTo (0); // to record if the current foreground pixel has been looped in Neighbours()

        bool isGlobalUpdate;
        if (m_impl->mIsCoarseDetection)
        {
            m_impl->SgMaskCalculation (frame);

            m_impl->CoarseForegroundDetection (frame);

            isGlobalUpdate = !(m_impl->mFrameCount%m_impl->m_param.GlobalUpdateRate ());    // if true, do global update
        }
        else
        {
            isGlobalUpdate = true;
        }

        Size fsize = frame.size();
        int channels = frame.channels ();
        int windowSize = m_impl->m_param.WindowSize ();
        //int gridSize = m_impl->m_param.GridSize ();
        const uchar* pdata = frame.ptr<uchar>();
		const uchar* pregionMask = regionMask == nullptr ? nullptr : regionMask->ptr<uchar>();
        uchar* pUpdateMask = m_impl->mpUpdateMask->ptr<uchar>();

        for (int i = 0, n = 0; i < fsize.height; ++i)
        {
            for (int j = 0; j < fsize.width; ++j, ++n)
            {
                //int m = i/gridSize*fsize.width/gridSize + j/gridSize;   /// corresponding position in the coarse image

                // selective update pixel information  updateMask regions + global update rate + if in initialisation stage update all
                if (m_impl->mFrameCount < windowSize || isGlobalUpdate || pUpdateMask[n])
                {
                    int p = n*channels;
                    Impl::CGmmPixelInfo* pPixelInfo = &m_impl->mpPixelInfo[n];

                    float pixel[CGmmBgModel::GMM_DIMENSION] = {0.0f};
                    for (int d = 0; d < channels; ++d)
                    {
                        pixel[d] = static_cast<float>(pdata[p+d]);      /// extract pixel value
                    }

                    int matchIndex[CGmmBgModel::GMM_MAX_NUMBER] = {0};
                    // test if current pixel matches any Gaussian in the model, if has match isMatch == true and matchIndex[] set the corresponding match position to 1
                    bool isMatch = m_impl->GmmMatchTest (pixel, pPixelInfo, matchIndex, channels);
                        
                    if (isMatch)
                        m_impl->Gm_mupdateMatch (pixel, pPixelInfo, matchIndex, isGlobalUpdate, pUpdateMask[n], m_impl->mFrameCount, channels);
                    else
                        m_impl->Gm_mupdateNoMatch (pixel, pPixelInfo, m_impl->mFrameCount, channels);

                    // sorting index for sorting each gaussian after update (in descending order according to weight/Sigma2)
                    m_impl->GmmSort (pPixelInfo, matchIndex, channels);

                    // generate foreground mask at each pixel position 
                    m_impl->GmmForegourndPixel(pPixelInfo, matchIndex, n, pregionMask == nullptr ? 255 : pregionMask[n]);
                        
                }
            }
        }

        m_impl->GmmBackground ();

        m_impl->mFrameCount++;

    }



    // Remove shadows and highlights
    void CGmmBgModel::ShadowHighlightRemoval(const Mat& frame)
    {
        Size fsize = frame.size();
        int channels = frame.channels ();
        int gridSize = m_impl->m_param.GridSize ();
        float shadowBrightThres = m_impl->m_param.ShadowBrightThreshold ();
        float shadowChromaThres = m_impl->m_param.ShadowChromaThreshold ();
        float highightThreshold = m_impl->m_param.HighlightThreshold ();
        const uchar* pdata = frame.ptr<uchar>();
        uchar* pForeground = m_impl->mpForeground->ptr<uchar> ();
        uchar* pSgMask = NULL;
        if (m_impl->mIsCoarseDetection)
            pSgMask = m_impl->mpSgMask->ptr<uchar>();

        for (int i = 0, n = 0; i < fsize.height; ++i)
        {
            for (int j = 0; j < fsize.width; ++j, ++n)
            {
                int p = n*channels;
                int m = i/gridSize*fsize.width/gridSize + j/gridSize;

                // per pixel based shadow detection
                if (pForeground[n] != 0)
                {
                    float sumTemp1 = 0.0f;
                    float sumTemp2 = 0.0f;
                    float sum_sigma2 = 0.0f;

                    for (int d = 0; d < channels; ++d)
                    {
                        sumTemp1 += pdata[p+d]*m_impl->mpPixelInfo[n].m_info[0].m_mu[d]/m_impl->mpPixelInfo[n].m_info[0].m_sigma2[d];
                        sumTemp2 += m_impl->mpPixelInfo[n].m_info[0].m_mu[d]*m_impl->mpPixelInfo[n].m_info[0].m_mu[d]/m_impl->mpPixelInfo[n].m_info[0].m_sigma2[d];
                        sum_sigma2 += m_impl->mpPixelInfo[n].m_info[0].m_sigma2[d];
                    }

                    float brightDistortion = sumTemp1/sumTemp2;
                    float chromaDistrotion = 0.0f;
                    for (int d = 0; d < channels; ++d)
                    {
                        chromaDistrotion += (pdata[p+d] - brightDistortion*m_impl->mpPixelInfo[n].m_info[0].m_mu[d])*(pdata[p+d] - brightDistortion*m_impl->mpPixelInfo[n].m_info[0].m_mu[d]);
                    }

                    // removal shadow and highlight
                    if (m_impl->mIsCoarseDetection)
                    {
                        if (brightDistortion > shadowBrightThres && brightDistortion < highightThreshold && chromaDistrotion < shadowChromaThres*shadowChromaThres*sum_sigma2)
                        {
                            pForeground[n] = 0;
                        } else if (brightDistortion > highightThreshold && chromaDistrotion < shadowChromaThres*shadowChromaThres*sum_sigma2 && pSgMask[m] == 0)
                        {
                            pForeground[n] = 0;
                        }
                    }
                    else
                    {
                        if (brightDistortion > shadowBrightThres && brightDistortion < highightThreshold && chromaDistrotion < shadowChromaThres*shadowChromaThres*sum_sigma2)
                        {
                            pForeground[n] = 0;
                        }
                    }

                }
            }
        }
    }

    // Remove small regions and fill holes
    void CGmmBgModel::ForegroundNoiseRemoval()
    {
        // foreground median filtering
        medianBlur(*m_impl->mpForeground, *m_impl->mpForeground, m_impl->m_param.MedfiltSize ());

        //// dilate foreground to include more boundary
        //dilate(*m_impl->mpForeground, *m_impl->mpForeground, Mat());

        // find each separated foreground region using findContours
        //findContours (*m_impl->mpForeground, m_impl->mFgContours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
        findContours (*m_impl->mpForeground, m_impl->mFgContours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

            
        m_impl->mpTempMat->setTo (0); // to record if the current foreground pixel has been looped in Neighbours()
        //FindForegroundRegions(*m_impl->mpForeground, m_impl->mFgRegions);

		// approximate contours to polygons + get bounding rects
		vector<vector<Point> > contoursPoly(m_impl->mFgContours.size());
		//vector<Rect> boundRect(m_impl->mFgContours.size());
		vector<Rect>().swap(m_impl->mFgBoxes);
		m_impl->mFgBoxes.reserve(m_impl->mFgContours.size());
		for (size_t i = 0; i < m_impl->mFgContours.size(); ++i)
		{
			approxPolyDP(Mat(m_impl->mFgContours[i]), contoursPoly[i], 3, true);
			m_impl->mFgBoxes.push_back(boundingRect(Mat(contoursPoly[i])));
		}


        // delete small regions
        //m_impl->mFgRegions.erase (remove_if(m_impl->mFgRegions.begin (), m_impl->mFgRegions.end(), LessRegion(m_impl->m_param.MinRegionWidth (), m_impl->m_param.MinRegionHeight ())), m_impl->mFgRegions.end());


        //vector<vector<Point> >::iterator it = m_impl->mFgRegions.begin ();
        //while (it != m_impl->mFgRegions.end())
        //{
        //    if (static_cast<int>((*it).size()) < m_impl->m_param.MinArea())
        //        it = m_impl->mFgRegions.erase(it);
        //    else
        //        ++it;
        //}

        //FindForegroundBoxes(m_impl->mFgRegions, m_impl->mFgBoxes);

        // delete small regions
        //vector<vector<Point> >::iterator itP = m_impl->mFgRegions.begin ();
        vector<Rect>::iterator itR = m_impl->mFgBoxes.begin ();
        while (itR != m_impl->mFgBoxes.end())
        {
            if (static_cast<int>((*itR).width) < m_impl->m_param.MinRegionWidth () || static_cast<int>((*itR).height) < m_impl->m_param.MinRegionHeight ())
            {
                itR = m_impl->mFgBoxes.erase(itR);
                //itP = m_impl->mFgRegions.erase (itP);
            }
            else
            {
                ++itR;
                //++itP;
            }
        }

		// find FgRegions
		//BOOST_FOREACH(const Rect& rect, m_impl->mFgBoxes)
		//{
		//	for (int x = rect.tl().x; x <= rect.br().x; ++x)
		//		for (int y = rect.tl().y; y <= rect.br().y; ++y)
		//			if (m_impl->mpForeground->
		//}
    }

    void CGmmBgModel::ForegroundFillHoles ()
    {
        m_impl->mpForeground->setTo(0);   // reset foreground

        // fill holes within each foreground region
        drawContours (*m_impl->mpForeground, m_impl->mFgContours, -1, Scalar(255, 255, 255), CV_FILLED);

        Mat foreground;
        m_impl->mpForeground->copyTo (foreground);
        //// find the filled regions  ????? IT IS CONTOURS NOT WHOLE PIXELS WITHIN REGIONS
        //findContours (foreground, m_impl->mFgRegions, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

    }

    void CGmmBgModel::BgModelPostUpdate (const Mat& frame)
    {
		//if(FrameCount() > 50)
  //      {
			ShadowHighlightRemoval (frame);
			ForegroundNoiseRemoval ();
			ForegroundFillHoles ();
		//}
    }

    void CGmmBgModel::FindForegroundRegions (const cv::Mat& foreground, std::vector<std::vector<cv::Point> >& regions)
    {
        Size fsize = foreground.size();
        regions.clear ();
        for (int i = 0; i < fsize.width*fsize.height; ++i)
        {
            vector<Point> points;
            m_impl->Neighbours (foreground, points, i, i);

            if (points.size() > 0)
            {
                regions.push_back (points);
            }
        }
        vector<vector<Point> >(regions).swap(regions);  // crop capacity
    }


    void CGmmBgModel::FindForegroundBoxes(std::vector<std::vector<cv::Point> >& regions, std::vector<cv::Rect>& boxes)
    {
        vector<Rect>().swap(boxes);    // clear and reduce boxes' capacity
        boxes.reserve(regions.size());
        for (size_t i = 0; i < regions.size(); ++i)
        {
            Point tl(100000, 100000);
            Point br(0,0);

            for (size_t j = 0; j < regions[i].size(); ++j)
            {
                tl.x = tl.x < regions[i][j].x ? tl.x : regions[i][j].x;
                tl.y = tl.y < regions[i][j].y ? tl.y : regions[i][j].y;
                br.x = br.x > regions[i][j].x ? br.x : regions[i][j].x;
                br.y = br.y > regions[i][j].y ? br.y : regions[i][j].y;
            }
            Rect temp(tl, br);

            boxes.push_back(temp);
        }
    }


    void CGmmBgModel::GetForegroundPoints(const std::vector<std::vector<cv::Point> >& regions, std::vector<cv::Point2f>& points)
    {
        vector<Point2f>().swap (points);
        for (size_t i = 0; i < regions.size(); ++i)
        {
            /*for (cv::vector<cv::Point>::const_iterator it = regions[i].begin(); it != regions[i].end(); ++it)
            {
                points.push_back (*it);
            }*/

            vector<Point>::size_type exsize = regions[i].size();
            size_t psize = points.size();
            points.resize(psize+exsize);
            copy(regions[i].begin(), regions[i].end(), points.begin()+psize);
        }

    }


	bool CGmmBgModel::IsForgroundOverCrowded() const
	{
		return static_cast<float>(cv::sum(*m_impl->mpForeground)[0]) / (m_impl->mpForeground->size().height * m_impl->mpForeground->size().width * 255) > 0.25;
	}


    const long CGmmBgModel::FrameCount() const
    {
        return m_impl->mFrameCount;
    }


    const Mat& CGmmBgModel::Background () const
    {
        return *(m_impl->mpBackground);
    }
    Mat& CGmmBgModel::Background ()
    {
        return *(m_impl->mpBackground);
    }

    const Mat& CGmmBgModel::Foreground () const
    {
        return *(m_impl->mpForeground);
    }
    Mat& CGmmBgModel::Foreground ()
    {
        return *(m_impl->mpForeground);
    }

    const Mat* CGmmBgModel::SgMask () const
    {
		if (m_impl->mpSgMask)
			return m_impl->mpSgMask.get();
		else
			return NULL;
    }
        
    const Mat* CGmmBgModel::FrameGray() const
    {
        if (m_impl->mpTempInfo)
            return m_impl->mpTempInfo->m_frameGray.get();
        else
            return NULL;
    }

    const Mat* CGmmBgModel::BackgroundGray () const
    {
        if (m_impl->mpTempInfo)
            return m_impl->mpTempInfo->m_backgroundGray.get();
        else
            return NULL;
    }


    const cv::Mat* CGmmBgModel::FrameGradientX() const
    {
        if (m_impl->mpTempInfo)
            return m_impl->mpTempInfo->m_frameGradientX.get();
        else
            return NULL;
    }

    const cv::Mat* CGmmBgModel::FrameGradientY() const
    {
        if (m_impl->mpTempInfo)
            return m_impl->mpTempInfo->m_frameGradientY.get();
        else
            return NULL;
    }

    const cv::Mat* CGmmBgModel::FrameGradientMag() const
    {
        if (m_impl->mpTempInfo)
            return m_impl->mpTempInfo->m_frameGradientMag.get();
        else
            return NULL;
    }


    const cv::Mat* CGmmBgModel::BackgroundGradientX() const
    {
        if (m_impl->mpTempInfo)
            return m_impl->mpTempInfo->m_backgroundGradientX.get();
        else
            return NULL;
    }

    const cv::Mat* CGmmBgModel::BackgroundGradientY() const
    {
        if (m_impl->mpTempInfo)
            return m_impl->mpTempInfo->m_backgroundGradientY.get();
        else
            return NULL;
    }

    const cv::Mat* CGmmBgModel::BackgroundGradientMag() const
    {
        if (m_impl->mpTempInfo)
            return m_impl->mpTempInfo->m_backgroundGradientMag.get();
        else
            return NULL;
    }

    const Mat* CGmmBgModel::CFMaskF() const
    {
        if (m_impl->mpTempInfo)
            return m_impl->mpCFMaskF.get();
        else
            return NULL;
    }
    const Mat* CGmmBgModel::UpdateMask() const
    {
        if (m_impl->mpTempInfo)
            return m_impl->mpUpdateMask.get();
        else
            return NULL;
    }

    //const vector<vector<Point> >& CGmmBgModel::ForegroundRegions() const
    //{
    //    return m_impl->mFgRegions;
    //}

    const vector<Rect>& CGmmBgModel::ForegroundBoxes () const
    {
        return m_impl->mFgBoxes;
    }

}