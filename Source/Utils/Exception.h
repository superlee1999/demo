#ifndef UTILS_EXCEPTION_H
#define UTILS_EXCEPTION_H

#include "Export.h"
#include <vector>
#include <exception>
#include <memory>

#pragma warning(disable: 4251)

class APT_API CException : public std::exception
{
public:
	CException(const char* msg) : _msg(msg) {}
	CException(const std::string& msg) : _msg(msg) {}
	virtual ~CException() throw() {}
	const char* what() const throw() override { return _msg.c_str(); }

private:
	std::string _msg;
};

#define CREQUIRE(expr, msg) { std::ostringstream os; os << msg; if(expr); else throw CException(os.str().c_str()); }
#define CEXCEPT(msg) { std::ostringstream os; os << msg; throw CException(os.str().c_str()); }

#endif