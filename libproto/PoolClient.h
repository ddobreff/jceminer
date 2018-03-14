/*  Blah, blah, blah.. all this pedantic nonsense to say that this
    source code is made available under the terms and conditions
    of the accompanying GNU General Public License */

#pragma once

#include <boost/asio/ip/address.hpp>

#include <libethcore/Farm.h>
#include <libethcore/Miner.h>
#include <libproto/PoolURI.h>

using namespace std;

namespace dev
{
namespace eth
{
class PoolConnection
{
public:
    PoolConnection() {};
    PoolConnection(const URI& uri)
    {
        m_uri = uri;
    };
    string Host() const
    {
        return m_uri.Host();
    };
    unsigned short Port() const
    {
        return m_uri.Port();
    };
    string User() const
    {
        return m_uri.User();
    };
    string Path() const
    {
        return m_uri.Path();
    };
    string Pass() const
    {
        return m_uri.Pswd();
    };
    SecureLevel SecLevel() const
    {
        return m_uri.ProtoSecureLevel();
    };
    boost::asio::ip::address Address() const
    {
        return m_address;
    };
    unsigned Version() const
    {
        return m_uri.ProtoVersion();
    };

    void Address(boost::asio::ip::address& address)
    {
        m_address = address;
    };

    bool Empty()
    {
        return m_uri.Empty();
    }

private:
    URI m_uri;
    boost::asio::ip::address m_address;
};

class PoolClient
{
public:
    void setConnection(PoolConnection& conn)
    {
        m_conn = conn;
        m_connection_changed = true;
    }

    virtual void connect() = 0;

    virtual void submitHashrate(uint64_t rate) = 0;
    virtual void submitSolution(Solution solution) = 0;
    virtual bool isConnected() = 0;

    using SolutionAccepted = std::function<void(bool const&)>;
    using SolutionRejected = std::function<void(bool const&, std::string const&)>;
    using Disconnected = std::function<void()>;
    using Connected = std::function<void(boost::asio::ip::address address)>;
    using WorkReceived = std::function<void(WorkPackage const&)>;

    void onSolutionAccepted(SolutionAccepted const& _handler)
    {
        m_onSolutionAccepted = _handler;
    }
    void onSolutionRejected(SolutionRejected const& _handler)
    {
        m_onSolutionRejected = _handler;
    }
    void onDisconnected(Disconnected const& _handler)
    {
        m_onDisconnected = _handler;
    }
    void onConnected(Connected const& _handler)
    {
        m_onConnected = _handler;
    }
    void onWorkReceived(WorkReceived const& _handler)
    {
        m_onWorkReceived = _handler;
    }

protected:
    bool m_authorized = false;
    PoolConnection m_conn;
    bool m_connection_changed = false;

    SolutionAccepted m_onSolutionAccepted;
    SolutionRejected m_onSolutionRejected;
    Disconnected m_onDisconnected;
    Connected m_onConnected;
    WorkReceived m_onWorkReceived;
};
}
}

