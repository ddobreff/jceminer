/*  Blah, blah, blah.. all this pedantic nonsense to say that this
    source code is made available under the terms and conditions
    of the accompanying GNU General Public License */

#include <map>
#include <boost/optional/optional_io.hpp>
#include <boost/algorithm/string.hpp>
#include <network/uri/detail/decode.hpp>
#include <libproto/PoolURI.h>

using namespace dev;

typedef struct {
    SecureLevel secure;
    unsigned version;
} SchemeAttributes;

static std::map<std::string, SchemeAttributes> s_schemes = {
    {"stratum+tcp",    {SecureLevel::NONE,  0}},
    {"ethproxy+tcp",   {SecureLevel::NONE,  1}},
    {"nicehash+tcp",   {SecureLevel::NONE,  2}},
    {"stratum+tls",    {SecureLevel::TLS,   0}},
    {"ethproxy+tls",   {SecureLevel::TLS,   1}},
    {"nicehash+tls",   {SecureLevel::TLS,   2}},
    {"stratum+tls12",  {SecureLevel::TLS12, 0}},
    {"ethproxy+tls12", {SecureLevel::TLS12, 1}},
    {"nicehash+tls12", {SecureLevel::TLS12, 2}}
};

URI::URI() {}

URI::URI(const std::string uri)
{
    m_uri = network::uri(uri);
}

bool URI::KnownScheme()
{
    std::string s(*m_uri.scheme());
    s = network::detail::decode(s);
    boost::trim(s);
    return s_schemes.find(s) != s_schemes.end();
}

unsigned URI::ProtoVersion() const
{
    std::string s(*m_uri.scheme());
    s = network::detail::decode(s);
    boost::trim(s);
    return s_schemes[s].version;
}

SecureLevel URI::ProtoSecureLevel() const
{
    std::string s(*m_uri.scheme());
    s = network::detail::decode(s);
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
    s = network::detail::decode(s);
    boost::trim(s);
    return s;
}

std::string URI::Host() const
{
    std::string s(*m_uri.host());
    s = network::detail::decode(s);
    boost::trim(s);
    if (s == "--")
        return "";
    return s;
}

unsigned short URI::Port() const
{
    std::string s(*m_uri.port());
    s = network::detail::decode(s);
    boost::trim(s);
    if (s == "--")
        return 0;
    return (unsigned short)atoi(s.c_str());
}

std::string URI::User() const
{
    std::string s(*m_uri.user_info());
    s = network::detail::decode(s);
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
    s = network::detail::decode(s);
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

