#ifndef UTILS_EXPORT_H
#define UTILS_EXPORT_H

#if defined(_WINDOWS) && !defined(WINDOWS)
	#define WINDOWS
#endif

#if defined(WIN32) || defined(WINDOWS)	// windows
	#ifdef APT_EXPORTS
		#define APT_API __declspec(dllexport)
	#else
		#define APT_API __declspec(dllimport)
	#endif
#else	// linux
	#ifdef APT_EXPORTS
		#define APT_API //__attribute__((visibility("default")))
	#else
		#define APT_API
	#endif
#endif

#endif