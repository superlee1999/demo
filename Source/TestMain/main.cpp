#include "Utils/CIString.h"
#include "Utils/Exception.h"
#include "Utils/CVROI.h"
#include "FG/Detection.h"
#include "Tracking/LinearAssignment.h"
#include "Tracking/Tracker.h"

#include <cstdint>
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>
#include <boost/algorithm/string/trim.hpp>


#define CAM_WIDTH 352
#define CAM_HEIGHT 288
#define CAM_CHANNELS 3
#define CV_FRAME_BUFFER_SIZE 50

using namespace std;
using namespace cv;
using namespace Utils;
using namespace FG;
using namespace Tracking;

const cv::Size CIF = cv::Size(CAM_WIDTH, CAM_HEIGHT);
const cv::Size CIF4 = cv::Size(CAM_WIDTH * 2, CAM_HEIGHT * 2);


int main(int argc, char* argv[])
{
	// extracting parameters from config file
	const char* configDir = getenv("APT_CONFIG_DIR");
	if (!configDir)
	{
		std::cerr << "Environment variable: APT_CONFIG_DIR is not set!" << endl;
		return -1;
	}

	string configFile = boost::trim_copy(string(configDir)) + "/cctv_config.json";

	VideoCapture cam;
	if (argc == 1)	// open webcam
	{
		if (!cam.open(0))
		{
			std::cerr << "Can't open webcam !!!" << std::endl;
			return -1;
		}
	}
	else if (argc == 2) // use video
	{
		string ipCamStr = argv[1];
		if(!cam.open(ipCamStr))
		{
			std::cerr << "Could not acquire image from " << ipCamStr << " !!!" << std::endl;
			return -1;
		}

	}

	// read in one frame and resize it
    Mat OriginalFrame;
	Mat frame;
	if(!cam.read(OriginalFrame))
	{
		std::cout << "Image capturing failed!" << std::endl;
		return -1;
	}
	resize(OriginalFrame, frame, CIF);

	int sampleFactor = 1;
	string windowName1 = "Tracking";
	string windowName2 = "FG";
	string windowName3 = "Backgournd";
	// define ROI in the frame
	namedWindow(windowName1);
	namedWindow(windowName2);
	namedWindow(windowName3);
	std::string display = "Define ROI, press Enter to exit";
	static CVROI::SPass2Mouse param(frame, windowName1);
	while (true)
	{
		// read in and resize frame
		for (int i = 0; i < sampleFactor; ++i)
		{
			if(!cam.read(OriginalFrame))
			{
				std::cerr << "Image capturing failed!" << std::endl;
				return -1;
			}
		}
		resize(OriginalFrame, frame, CIF);

		int key = waitKey(10);
		switch (key)
		{
		case 49:	// 1
			param.m_colour = CVROI::EColour::RED;
			break;
		case 50:	// 2
			param.m_colour = CVROI::EColour::GREEN;
			break;
		case 51:	// 3 
			param.m_colour = CVROI::EColour::BLUE;
			break;
		case 52:	// 4
			param.m_colour = CVROI::EColour::YELLOW;
			break;
		case 53:	// 5
			param.m_colour = CVROI::EColour::CYAN;
			break;
		case 54:	// 6
			param.m_colour = CVROI::EColour::MAGENTA;
			break;
		}

		param.m_img = frame;
		setMouseCallback(windowName1, CVROI::onMouse, &param);

		CVROI::DrawBoxes(&frame, param.m_rois);
		putText(frame, display, Point(10, frame.size().height-10), FONT_HERSHEY_PLAIN, 1.0, CVROI::Colour(CVROI::EColour::BLUE));
		cv::imshow(windowName1, frame);
		if (key == 13 || key == 1113997)  // "Enter" to end loop
        {
            break;
        }

	}

	// create a foreground region mask based on the regions selected from the settings stage
	cv::Mat regionMask(frame.size().height, frame.size().width, CV_8UC1);
	regionMask.setTo(0);
	std::vector<cv::Rect> sceneExits;
	bool hasRegionsDefined = false;
	for (const auto& roi: param.m_rois)
	{
		if (roi.m_colour == CVROI::EColour::MAGENTA)
			sceneExits.push_back(roi.m_rect);
		else
		{
			cv::Mat subMat = regionMask.colRange(roi.m_rect.tl().x, roi.m_rect.br().x).rowRange(roi.m_rect.tl().y, roi.m_rect.br().y);
			subMat.setTo(255);
			hasRegionsDefined = true;
		}
	}

	if (!hasRegionsDefined)
		regionMask.setTo(255);

	std::unique_ptr<CGmmBgModel> bgModel;
	std::unique_ptr<Tracker::CMultiTracker> tracker;

	bool hasInit = false;
	while (true)
	{
		// read in and resize frame
		for (int i = 0; i < sampleFactor; ++i)
		{
			if(!cam.read(OriginalFrame))
			{
				std::cerr << "Image capturing failed!" << std::endl;
				return -1;
			}
		}
		resize(OriginalFrame, frame, CIF);   
		
		if (!hasInit)
		{
			bgModel.reset(new CGmmBgModel(frame, CGmmBgModelParam(configFile), false));
			tracker.reset(new Tracker::CMultiTracker(configFile));
			hasInit = true;
		}
		else
		{
			bgModel->BgModelUpdate(frame, &regionMask);
			bgModel->BgModelPostUpdate(frame);

			if (bgModel->IsForgroundOverCrowded())
				tracker->UpdateTracks(std::vector<cv::Rect>(), frame, nullptr, sceneExits.empty() ? nullptr : &sceneExits);
			else
				tracker->UpdateTracks(bgModel->ForegroundBoxes(), frame, &bgModel->Foreground(), sceneExits.empty() ? nullptr : &sceneExits);

			tracker->DisplayTracks(frame, cv::Size(5, 5));
		}

		CVROI::DrawBoxes(&frame, param.m_rois);
		cv::imshow("Tracking", frame);
		cv::imshow("FG", bgModel->Foreground());
		cv::imshow("Background", bgModel->Background());
        if (cv::waitKey(10) >= 0)
		{
			break;
		}
	}

	cv::destroyAllWindows();


	//LOG_INFO("End of IP cam processing!");

    return 0;
}
