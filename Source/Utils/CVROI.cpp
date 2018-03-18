#include "CVROI.h"
#include <boost/assign.hpp>


using namespace std;
using namespace cv;
using namespace boost;

namespace {
	    /// A class for recording box's top-left and bottom-right corners
        struct SBoxStartEndPoints
        {
			SBoxStartEndPoints()
				: m_start(-1, -1)
				, m_end(-1, -1)
			{}

            void reset()
            {
                m_start.x = -1;
                m_start.y = -1;
                m_end.x = -1;
                m_end.y = -1;
            }
            cv::Point m_start;
            cv::Point m_end;
        };
}


namespace Utils {
	namespace CVROI {

		//std::vector<SRectRoi> SPass2Mouse::m_rois;


		cv::Scalar Colour(EColour colour)
		{
			static std::map<EColour, cv::Scalar> colourMap = boost::assign::map_list_of
				(EColour::RED, cv::Scalar(0, 0, 255))
				(EColour::GREEN, cv::Scalar(0, 255, 0))
				(EColour::BLUE, cv::Scalar(255, 0, 0))
				(EColour::YELLOW, cv::Scalar(0, 255, 255))
				(EColour::CYAN, cv::Scalar(255, 255, 0))
				(EColour::MAGENTA, cv::Scalar(255, 0, 255));

			return colourMap[colour];
		}


		void onMouse(int event, int x, int y, int flags, void* param)
        {
            SPass2Mouse* pMyParam = static_cast<SPass2Mouse*>(param);

            // boundary check
            Size fsize = pMyParam->m_img.size();
            if (x < 0)
                x = 0;
            if (x > fsize.width)
                x = fsize.width-1;
            if (y < 0)
                y = 0;
            if (y > fsize.width)
                y = fsize.height-1;

            static SBoxStartEndPoints points;

            switch (event)
            {
            case CV_EVENT_LBUTTONDOWN:  //record starting point
				{
					points.m_start.x = x;
					points.m_start.y = y;
					break;
				}
            case CV_EVENT_LBUTTONUP:    //record ending point
				{
					if (points.m_start.x >= 0 && points.m_start.y >= 0)
					{
						points.m_end.x = x;
						points.m_end.y = y;

						cv::Rect tempBox;
						tempBox.x = points.m_start.x < points.m_end.x ? points.m_start.x : points.m_end.x;
						tempBox.y = points.m_start.y < points.m_end.y ? points.m_start.y : points.m_end.y;
						tempBox.width = abs(points.m_end.x - points.m_start.x);
						tempBox.height = abs(points.m_end.y - points.m_start.y);

						pMyParam->m_rois.push_back(std::move(SRectRoi(tempBox, pMyParam->m_colour, 1)));
					}
					break;
				}
            case CV_EVENT_MOUSEMOVE:    // draw intermediate box's track
				{
					if (flags & CV_EVENT_FLAG_LBUTTON)
					{
						points.m_end.x = x;
						points.m_end.y = y;

						//pMyParam->m_img.copyTo (pMyParam->m_imgCopy);
						DrawBoxes (&pMyParam->m_img, pMyParam->m_rois);
						if (points.m_start.x >= 0 && points.m_start.y >= 0)
						{
							rectangle (pMyParam->m_img, points.m_start, points.m_end, Colour(pMyParam->m_colour));
						}
						imshow (pMyParam->m_windowName, pMyParam->m_img);
					}
					break;
				}

            case CV_EVENT_RBUTTONDOWN:  // delete bounding box
				{
					auto deleteIter = std::find_if(pMyParam->m_rois.begin(), pMyParam->m_rois.end(), [x, y](const SRectRoi& roi){ return roi.m_rect.contains(Point(x, y)); });
                
					if (deleteIter != pMyParam->m_rois.end())
					{
						pMyParam->m_rois.erase(deleteIter);
						//pMyParam->m_img.copyTo (pMyParam->m_imgCopy);
						DrawBoxes (&pMyParam->m_img, pMyParam->m_rois);
						imshow (pMyParam->m_windowName, pMyParam->m_img);
					}
					else
					{
						if (!pMyParam->m_rois.empty())
						{
							pMyParam->m_rois.pop_back();
							//pMyParam->m_img.copyTo (pMyParam->m_imgCopy);
							DrawBoxes (&pMyParam->m_img, pMyParam->m_rois);
							imshow (pMyParam->m_windowName, pMyParam->m_img);
						}
					}

					break;
				}
            }
        }

		void DrawBoxes(cv::Mat* image, const std::vector<SRectRoi>& rois)
		{
			for (auto& roi: rois)
			{
				rectangle(*image, roi.m_rect.tl(), roi.m_rect.br(), Colour(roi.m_colour), roi.m_thickness);
			}
		}
	}
}