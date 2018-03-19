/*  Blah, blah, blah.. all this pedantic nonsense to say that this
    source code is made available under the terms and conditions
    of the accompanying GNU General Public License */

#pragma once

#include <network/uri.hpp>
#include <string>

// A simple URI parser specifically for mining pool enpoints
namespace dev
{

enum class SecureLevel {NONE = 0, TLS12, TLS};

class URI : network::uri
{
public:
	URI();
	URI(const std::string uri);

	std::string	Scheme() const;
	std::string	Host() const;
	unsigned short	Port() const;
	std::string	User() const;
	std::string	Pswd() const;
	SecureLevel	ProtoSecureLevel() const;
	unsigned	ProtoVersion() const;

	bool		KnownScheme();
	bool		Empty();

	static std::string KnownSchemes(SecureLevel secureLevel);

private:
	network::uri   m_uri;
};

}
