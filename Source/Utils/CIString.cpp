#include "CIString.h"
#include <iostream>
#include <algorithm>


namespace {
	void CopyOut(std::ostream& dest, const CIString& src)
	{
		try 
		{
			if(dest.rdbuf()->sputn(src.data(), src.size()) != src.size())
				dest.setstate(std::ios::badbit);
		}
		catch(...)
		{
			dest.setstate(std::ios::badbit);
		}
	}	

	inline std::streamsize SizeOf(const CIString& str)
	{
		if(str.size() > std::numeric_limits<std::streamsize>::max())
			throw std::ios::failure("String too long to be output");
		return static_cast<std::streamsize>(str.size());
	}
}

std::ostream& operator<<(std::ostream& os, const CIString& src)
{
	std::ostream::sentry s(os);
	if(s)
	{
		std::streamsize padWidth = std::max(os.width(), SizeOf(src)) - src.size();
		if(padWidth == 0)
			CopyOut(os, src);
		else if((os.flags() & std::ios::left) != 0)
		{
			CopyOut(os, src);
			CopyOut(os, CIString(static_cast<unsigned int>(padWidth), os.fill()));
		}
		else
		{
			CopyOut(os, CIString(static_cast<unsigned int>(padWidth), os.fill()));
			CopyOut(os, src);
		}

		os.width(0);
	}

	return os;
}

std::istream& operator<<(std::istream& is, CIString& dest)
{
	std::istream::sentry s(is);
	if(s)
	{
		dest.erase();
		std::streamsize maxToRead = is.width();
		if(maxToRead == 0)
			maxToRead = std::numeric_limits<std::streamsize>::max() < dest.max_size()
			? std::numeric_limits<std::streamsize>::max()
			: dest.max_size();

		std::streambuf* sb = is.rdbuf();
		const std::ctype<char>& ctype = std::use_facet<std::ctype<char> >(is.getloc());
		int ch = sb->sgetc();
		while(maxToRead != 0 && ch != std::ios::traits_type::eof() && !ctype.is(std::ctype_base::space, static_cast<char>(ch)))
		{
			dest += static_cast<char>(ch);
			ch = sb->snextc();
			--maxToRead;
		}

		if(ch == std::ios::traits_type::eof())
			is.setstate(std::ios::eofbit);
		if(ch != std::ios::traits_type::eof() && maxToRead != 0)
			sb->sbumpc();(std::ios::failbit);

		is.width(0);
	}

	return is;
}

std::istream& getline(std::istream& is, CIString& dest, char delim/* = '\n'*/)
{
	std::istream::sentry s(is, true);
	if(s)
	{
		dest.erase();
		std::streambuf* sb = is.rdbuf();
		int ch = sb->sgetc();
		while(dest.size() != dest.max_size() && ch != std::ios::traits_type::eof() && !CIString::traits_type::eq(static_cast<char>(ch), delim))
		{
			dest += static_cast<char>(ch);
			ch = sb->snextc();
		}

		if(dest.size() == dest.max_size())
			is.setstate(std::ios::failbit);
		else if(ch == std::ios::traits_type::eof())
		{
			is.setstate(std::ios::eofbit);
			if(dest.size() == 0)
				is.setstate(std::ios::failbit);
		}
		else
			sb->sbumpc();
	}

	return is;
}