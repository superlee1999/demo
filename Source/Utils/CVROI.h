#ifndef UTILS_CVROI_H
#define UTILS_CVROI_H

#include "Export.h"
#include <vector>
#include <map>
#include <string>
#include <opencv2/opencv.hpp>

namespace Utils {
	namespace CVROI {

		enum class EColour
		{
			RED,
			GREEN,
			BLUE,
			YELLOW,
			CYAN,
			MAGENTA
		};

		APT_API cv::Scalar Colour(EColour colour = EColour::RED);

		struct APT_API SRectRoi
		{
			SRectRoi(): m_rect(), m_colour(EColour::RED), m_thickness(1) {}
			SRectRoi(const cv::Rect& rect, EColour colour, int thickness)
				: m_rect(rect)
				, m_colour(colour)
				, m_thickness(thickness)
			{
			}

			cv::Rect m_rect;
			EColour m_colour;
			int m_thickness;
		};

        /// Type to be passed to onMouse event
        struct APT_API SPass2Mouse
        {
			SPass2Mouse(const cv::Mat& image, const std::string& window, EColour colour = EColour::RED)
				: m_img(image)
				, m_windowName(window)
				, m_colour(colour)
            {
                //m_img.copyTo (m_imgCopy);
            }

            cv::Mat             m_img;			/// original image
            //cv::Mat             m_imgCopy;		/// copied image for rendering intermediate plot
            std::string         m_windowName;   /// name of active window
			EColour				m_colour;
            std::vector<SRectRoi>   m_rois;         /// rois drawn on image
        };

        /// Mouse callback function
        APT_API void onMouse(int event, int x, int y, int flags, void* param);
		APT_API void DrawBoxes(cv::Mat* image, const std::vector<SRectRoi>& rois);

	}
}




#endif