#ifndef UDP_EXPORT_H
#define UDP_EXPORT_H

#ifdef _WIN32
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT
#endif

#endif