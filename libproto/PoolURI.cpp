/*      This program is free software: you can redistribute it and/or modify
        it under the terms of the GNU General Public License as published by
        the Free Software Foundation, either version 3 of the License, or
        (at your option) any later version.

        This program is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        GNU General Public License for more details.

        You should have received a copy of the GNU General Public License
        along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <map>
#include <boost/optional/optional_io.hpp>
#include <boost/algorithm/string.hpp>
#include <libproto/PoolURI.h>

using namespace dev;

typedef struct {
	SecureLevel secure;
	unsigned version;
} SchemeAttributes;

static std::map<std::string, SchemeAttributes> s_schemes = {
	{"stratum+tcp",	   {SecureLevel::NONE,  0}},
	{"stratum1+tcp",   {SecureLevel::NONE,  1}},
	{"stratum2+tcp",   {SecureLevel::NONE,  2}},
	{"stratum+tls",	   {SecureLevel::TLS,   0}},
	{"stratum1+tls",   {SecureLevel::TLS,   1}},
	{"stratum2+tls",   {SecureLevel::TLS,   2}},
	{"stratum+tls12",  {SecureLevel::TLS12, 0}},
	{"stratum1+tls12", {SecureLevel::TLS12, 1}},
	{"stratum2+tls12", {SecureLevel::TLS12, 2}},
	{"stratum+ssl",	   {SecureLevel::TLS12, 0}},
	{"stratum1+ssl",   {SecureLevel::TLS12, 1}},
	{"stratum2+ssl",   {SecureLevel::TLS12, 2}}
};

URI::URI() {}

URI::URI(const std::string uri)
{
	m_uri = network::uri(uri);
}

bool URI::KnownScheme()
{
	std::string s(*m_uri.scheme());
	boost::trim(s);
	return s_schemes.find(s) != s_schemes.end();
}

unsigned URI::ProtoVersion() const
{
	std::string s(*m_uri.scheme());
	boost::trim(s);
	return s_schemes[s].version;
}

SecureLevel URI::ProtoSecureLevel() const
{
	std::string s(*m_uri.scheme());
	boost::trim(s);
	return s_schemes[s].secure;
}

std::string URI::KnownSchemes(SecureLevel secureLevel)
{
	std::string schemes;
	for (const auto& s : s_schemes)
		if (s.second.secure == secureLevel)
			schemes += s.first + " ";
	boost::trim(schemes);
	return schemes;
}

std::string URI::Scheme() const
{
	std::string s(*m_uri.scheme());
	boost::trim(s);
	return s;
}

std::string URI::Host() const
{
	std::string s(*m_uri.host());
	boost::trim(s);
	if (s == "--")
		return "";
	return s;
}

unsigned short URI::Port() const
{
	std::string s(*m_uri.port());
	boost::trim(s);
	if (s == "--")
		return 0;
	return (unsigned short)atoi(s.c_str());
}

std::string URI::User() const
{
	std::stringstream ss;
	std::string s(*m_uri.user_info());
	boost::trim(s);
	if (s == "--")
		return "";
	size_t f = s.find(":");
	if (f == std::string::npos)
		return s;
	return s.substr(0, f);
}

std::string URI::Pswd() const
{
	std::string s(*m_uri.user_info());
	boost::trim(s);
	if (s == "--")
		return "";
	size_t f = s.find(":");
	if (f == std::string::npos)
		return "";
	return s.substr(f + 1);
}

bool URI::Empty()
{
	return m_uri.empty();
}

