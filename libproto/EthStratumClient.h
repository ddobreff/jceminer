/*  Blah, blah, blah.. all this pedantic nonsense to say that this
    source code is made available under the terms and conditions
    of the accompanying GNU General Public License */

#pragma once

#include <iostream>
#include <boost/array.hpp>
#include <boost/asio.hpp>
#include <boost/asio/ssl.hpp>
#include <boost/bind.hpp>
#include <boost/atomic.hpp>
#include <json/json.h>
#include <libdevcore/bounded_queue.h>
#include <libdevcore/FixedHash.h>
#include <libethcore/Farm.h>
#include <libethcore/EthashAux.h>
#include <libethcore/Miner.h>
#include "PoolClient.h"


using namespace std;
using namespace dev;
using namespace dev::eth;

class EthStratumClient : public PoolClient
{
public:

    typedef enum { STRATUM = 0, ETHPROXY, ETHEREUMSTRATUM } StratumProtocol;

    EthStratumClient();
    ~EthStratumClient();

    void connect();
    bool isConnected()
    {
        return m_connected && m_authorized;
    }

    void submitHashrate(uint64_t rate);
    void submitSolution(Solution solution);

    h256 currentHeaderHash()
    {
        return m_current.header;
    }
    bool current()
    {
        return static_cast<bool>(m_current);
    }

private:

    void resolve_handler(const boost::system::error_code& ec, boost::asio::ip::tcp::resolver::iterator i);
    void connect_handler(const boost::system::error_code& ec, boost::asio::ip::tcp::resolver::iterator i);
    void work_timeout_handler(const boost::system::error_code& ec);
    void response_timeout_handler(const boost::system::error_code& ec);
    void stop_timeout_handler(const boost::system::error_code& ec);
    void hr_timeout_handler(const boost::system::error_code& ec);
    void reset_work_timeout();

    void readline();
    void handleResponse(const boost::system::error_code& ec);
    void handleHashrateResponse(const boost::system::error_code&) {};
    void handleSubmitResponse(const boost::system::error_code& ec, void* buf);
    void readResponse(const boost::system::error_code& ec, std::size_t bytes_transferred);
    void processReponse(Json::Value& responseObject);
    void async_write_with_response(boost::asio::streambuf& buff);

    PoolConnection m_connection;

    string m_worker; // eth-proxy only;

    bool m_authorized;
    bool m_connected = false;

    std::mutex x_pending;
    int m_pending;

    WorkPackage m_current;

    bool m_stale = false;

    std::thread m_serviceThread;  ///< The IO service thread.
    boost::asio::io_service m_io_service;
    boost::asio::ip::tcp::socket* m_socket;
    boost::asio::ssl::stream<boost::asio::ip::tcp::socket>* m_securesocket;

    boost::asio::streambuf m_requestBuffer;
    boost::asio::streambuf m_responseBuffer;
    boost::asio::streambuf m_hrBuffer;

    boost::asio::deadline_timer m_worktimer;
    boost::asio::deadline_timer m_responsetimer;
    boost::asio::deadline_timer m_stoptimer;
    boost::asio::deadline_timer m_hrtimer;
    bool m_response_pending = false;

    boost::asio::ip::tcp::resolver m_resolver;

    string m_email;

    double m_nextWorkDifficulty;

    h64 m_extraNonce;
    int m_extraNonceHexSize;

    bool m_submit_hashrate = false;
    string m_submit_hashrate_id;

    void processExtranonce(std::string& enonce);

    bool m_linkdown = true;
    uint64_t m_rate;

    tp::BoundedQueue<void*> m_freeBuffers;
};
