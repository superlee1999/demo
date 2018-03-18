#ifndef UTILS_CISTRING_H
#define UTILS_CISTRING_H

#include "Export.h"

#include <cstring>
#include <string>
#include <functional>
#include <iosfwd>

struct ci_char_traits: public std::char_traits<char>
{
	static bool eq(char left, char right) { return toupper(left) == toupper(right); }
	static bool lt(char left, char right) { return toupper(left) < toupper(right); }
	static int compare(const char* left, const char* right, size_t len) { return _memicmp(left, right, len); }
	static const char* find(const char* str, int len, char val)
	{
		while(len-- > 0 && toupper(*str) != toupper(val))
			++str;

		return len >= 0 ? str : NULL;
	}
};
typedef std::basic_string<char, ci_char_traits> CIString;

// IO support
APT_API std::ostream& operator<<(std::ostream& os, const CIString& src);
APT_API std::istream& operator<<(std::istream& is, CIString& dest);
APT_API std::istream& getline(std::istream& src, CIString& dest, char delim = '\n');

// comparison operators for the char in CIString. to be used in <algorithm>
struct ci_char_less: std::binary_function<char, char, bool>
{
	bool operator()(char lhs, char rhs) const
	{
		return ci_char_traits::lt(lhs, rhs);
	}
};

struct ci_char_equal: std::binary_function<char, char, bool>
{
	bool operator()(char lhs, char rhs)
	{
		return ci_char_traits::eq(lhs, rhs);
	}
};

#endif