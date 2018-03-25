/*  Blah, blah, blah.. all this pedantic nonsense to say that this
    source code is made available under the terms and conditions
    of the accompanying GNU General Public License */

#include "EthStratumClient.h"
#include "libethash/endian.h"
#include "libdevcore/Log.h"
#include <miner-buildinfo.h>

using boost::asio::ip::tcp;

extern bool g_report_stratum_hashrate;
extern string g_email;
extern unsigned g_worktimeout;

static void diffToTarget(uint32_t* target, double diff)
{
	uint32_t target2[8];
	uint64_t m;
	int k;

	for (k = 6; k > 0 && diff > 1.0; k--)
		diff /= 4294967296.0;
	m = (uint64_t)(4294901760.0 / diff);
	if (m == 0 && k == 6)
		memset(target2, 0xff, sizeof(target2));
	else {
		memset(target2, 0, sizeof(target2));
		target2[k] = (uint32_t)m;
		target2[k + 1] = (uint32_t)(m >> 32);
	}

	for (int i = 0; i < 32; i++)
		((uint8_t*)target)[31 - i] = ((uint8_t*)target2)[i];
}

extern unsigned g_stopAfter;

EthStratumClient::EthStratumClient() : PoolClient(),
	m_socket(nullptr),
	m_securesocket(nullptr),
	m_worktimer(m_io_service),
	m_responsetimer(m_io_service),
	m_stoptimer(m_io_service),
	m_hrtimer(m_io_service),
	m_resolver(m_io_service),
	m_submitBuffers(MAX_MINERS)
{
	m_authorized = false;
	m_pending = 0;
	m_submit_hashrate_id = h256::random().hex();
	if (g_stopAfter) {
		m_stoptimer.expires_from_now(boost::posix_time::minutes(g_stopAfter));
		m_stoptimer.async_wait(boost::bind(&EthStratumClient::stop_timeout_handler, this, boost::asio::placeholders::error));
	}
}

EthStratumClient::~EthStratumClient()
{
	m_io_service.stop();
	m_serviceThread.join();

	if (m_connection.SecLevel() != SecureLevel::NONE) {
		if (m_securesocket)
			delete m_securesocket;
	} else {
		if (m_socket)
			delete m_socket;
	}
}

void EthStratumClient::connect()
{
	m_connection = m_conn;

	m_authorized = false;
	m_connected = false;

	stringstream ssPort;
	ssPort << m_connection.Port();
	tcp::resolver::query q(m_connection.Host(), ssPort.str());

	if (m_connection.SecLevel() != SecureLevel::NONE) {

		boost::asio::ssl::context::method method = boost::asio::ssl::context::tls;
		if (m_connection.SecLevel() == SecureLevel::TLS12)
			method = boost::asio::ssl::context::tlsv12;

		boost::asio::ssl::context ctx(method);
		m_securesocket = new boost::asio::ssl::stream<boost::asio::ip::tcp::socket>(m_io_service, ctx);
		m_socket = &m_securesocket->next_layer();

		m_securesocket->set_verify_mode(boost::asio::ssl::verify_peer);

		char* certPath = getenv("SSL_CERT_FILE");
		try {
			ctx.load_verify_file(certPath ? certPath : "/etc/ssl/certs/ca-certificates.crt");
		} catch (std::exception const& e) {
			Guard l(x_log);
			logerror << e.what() << endl;
			logwarn << "Failed to load ca certificates. Either the file '/etc/ssl/certs/ca-certificates.crt' does not exist" <<
			        endl;
			logwarn << "or the environment variable SSL_CERT_FILE is set to an invalid or inaccessable file." << endl;
			logwarn << "It is possible that certificate verification can fail." << endl;
		}
	} else
		m_socket = new boost::asio::ip::tcp::socket(m_io_service);

	// Activate keep alive to detect disconnects
	unsigned int keepAlive = 10000;

	struct timeval tv;
	tv.tv_sec = keepAlive / 1000;
	tv.tv_usec = keepAlive % 1000;
	setsockopt(m_socket->native_handle(), SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
	setsockopt(m_socket->native_handle(), SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

	m_resolver.async_resolve(q, boost::bind(&EthStratumClient::resolve_handler, this,
	                                        boost::asio::placeholders::error, boost::asio::placeholders::iterator));

	if (m_serviceThread.joinable()) {
		// If the service thread have been created try to reset the service.
		m_io_service.reset();
	} else {
		// Otherwise, if the first time here, create new thread.
		m_serviceThread = std::thread{boost::bind(&boost::asio::io_service::run, &m_io_service)};
	}
}

#define BOOST_ASIO_ENABLE_CANCELIO

void EthStratumClient::resolve_handler(const boost::system::error_code& ec, tcp::resolver::iterator i)
{
	//dev::setThreadName("stratum");
	if (!ec) {
		tcp::resolver::iterator end;
		async_connect(*m_socket, i, end, boost::bind(&EthStratumClient::connect_handler, this,
		              boost::asio::placeholders::error, boost::asio::placeholders::iterator));
	} else {
		stringstream ss;
		{
			Guard l(x_log);
			logwarn << "Could not resolve host " << m_connection.Host() << ':' << m_connection.Port() << ", " << ec.message() <<
			        endl;
		}
		throw runtime_error("Can't resolve");
	}
}

void EthStratumClient::reset_work_timeout()
{
	m_worktimer.cancel();
	m_worktimer.expires_from_now(boost::posix_time::seconds(g_worktimeout));
	m_worktimer.async_wait(boost::bind(&EthStratumClient::work_timeout_handler, this, boost::asio::placeholders::error));
}

static void logJson(string json)
{
	Json::Value txObject;
	Json::Reader reader;
	reader.parse(json.c_str(), txObject);
	Guard l(x_log);
	loginfo << "JSON TX" << endl << txObject << endl;
}

void EthStratumClient::async_write_with_response(boost::asio::streambuf& buf)
{
	if (m_connection.SecLevel() != SecureLevel::NONE)
		async_write(*m_securesocket, buf,
		            boost::bind(&EthStratumClient::handleResponse, this, boost::asio::placeholders::error));
	else
		async_write(*m_socket, buf,
		            boost::bind(&EthStratumClient::handleResponse, this, boost::asio::placeholders::error));
}

void EthStratumClient::connect_handler(const boost::system::error_code& ec, tcp::resolver::iterator i)
{
	(void)i;

	//dev::setThreadName("stratum");

	if (!ec) {
		m_connected = true;
		m_linkdown = false;

		if (m_onConnected)
			m_onConnected(m_socket->remote_endpoint().address());

		if (m_connection.SecLevel() != SecureLevel::NONE) {
			boost::system::error_code hec;
			m_securesocket->handshake(boost::asio::ssl::stream_base::client, hec);
			if (hec) {
				{
					Guard l(x_log);
					logerror << "SSL/TLS Handshake failed: " << hec.message() << endl;
				}
				if (hec.value() == 337047686) { // certificate verification failed
					{
						Guard l(x_log);
						loginfo << "This can have multiple reasons:" << endl;
					}
					{
						Guard l(x_log);
						loginfo << "* Root certs are either not installed or not found" << endl;
					}
					{
						Guard l(x_log);
						loginfo << "* Pool uses a self-signed certificate" << endl;
					}
					{
						Guard l(x_log);
						loginfo << "Possible fixes:" << endl;
					}
					{
						Guard l(x_log);
						loginfo << "* Make sure the file '/etc/ssl/certs/ca-certificates.crt' exists and is accessable" << endl;
					}
					{
						Guard l(x_log);
						loginfo << "* Export the correct path via 'export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt' to the correct file"
						        << endl;
					}
					{
						Guard l(x_log);
						loginfo << "  On most systems you can install the 'ca-certificates' package" << endl;
					}
					{
						Guard l(x_log);
						loginfo << "  You can also get the latest file here: https://curl.haxx.se/docs/caextract.html" << endl;
					}
					{
						Guard l(x_log);
						loginfo << "* Disable certificate verification all-together via command-line option." << endl;
					}
				}
				throw runtime_error("No CA");
			}
		}

		// Successfully connected so we start our work timeout timer
		reset_work_timeout();

		std::stringstream json;

		string user;
		size_t p;

		switch (m_connection.Version()) {
		case EthStratumClient::STRATUM:
			m_authorized = true;
			json << "{\"id\": 1, \"method\": \"mining.subscribe\", \"params\": []}\n";
			break;
		case EthStratumClient::ETHPROXY:
			p = m_connection.User().find_first_of(".");
			user = m_connection.User().substr(0, p);
			if (p + 1 <= m_connection.User().length())
				m_worker = m_connection.User().substr(p + 1);
			else
				m_worker = "";

			if (m_email.empty())
				json << "{\"id\": 1, \"worker\":\"" << m_worker << "\", \"method\": \"eth_submitLogin\", \"params\": [\"" << user <<
				     "\"]}\n";
			else
				json << "{\"id\": 1, \"worker\":\"" << m_worker << "\", \"method\": \"eth_submitLogin\", \"params\": [\"" << user <<
				     "\", \"" << m_email << "\"]}\n";
			break;
		case EthStratumClient::ETHEREUMSTRATUM:
			m_authorized = true;
			json << "{\"id\": 1, \"method\": \"mining.subscribe\", \"params\": [\"miner/" <<
			     miner_get_buildinfo()->project_version << "\",\"EthereumStratum/1.0.0\"]}\n";
			break;
		}
		std::ostream os(&m_requestBuffer);
		os << json.str();
		async_write_with_response(m_requestBuffer);
		if (g_logJson)
			logJson(json.str());
	} else {
		{
			Guard l(x_log);
			logerror << "Could not connect to stratum server " << m_connection.Host() << ':' << m_connection.Port() << ", " <<
			         ec.message() << endl;
		}
		throw runtime_error("No connect");
	}

}

void EthStratumClient::readline()
{
	Guard l(x_pending);
	if (m_pending == 0) {
		if (m_connection.SecLevel() != SecureLevel::NONE)
			async_read_until(*m_securesocket, m_responseBuffer, "\n",
			                 boost::bind(&EthStratumClient::readResponse, this,
			                             boost::asio::placeholders::error, boost::asio::placeholders::bytes_transferred));
		else
			async_read_until(*m_socket, m_responseBuffer, "\n",
			                 boost::bind(&EthStratumClient::readResponse, this,
			                             boost::asio::placeholders::error, boost::asio::placeholders::bytes_transferred));
		m_pending++;
	}
}

void EthStratumClient::handleResponse(const boost::system::error_code& ec)
{
	if (!ec)
		readline();
	else {
		Guard l(x_log);
		logerror << "Handle response failed: " + ec.message() << endl;
	}
}

void EthStratumClient::handleSubmitResponse(const boost::system::error_code& ec)
{
	handleResponse(ec);
	boost::asio::streambuf* buf;
	if (m_submitBuffers.pop(buf))
		delete buf;
}

void EthStratumClient::readResponse(const boost::system::error_code& ec, std::size_t bytes_transferred)
{
	{
		Guard l(x_pending);
		m_pending = m_pending > 0 ? m_pending - 1 : 0;
	}

	if (!ec && bytes_transferred) {
		std::istream is(&m_responseBuffer);
		std::string response;
		getline(is, response);

		if (!response.empty() && response.front() == '{' && response.back() == '}') {
			Json::Value responseObject;
			Json::Reader reader;
			if (reader.parse(response.c_str(), responseObject))
				processReponse(responseObject);
			else {
				Guard l(x_log);
				logerror << "Parse response failed: " + reader.getFormattedErrorMessages() << endl;
			}
		} else if (m_connection.Version() != EthStratumClient::ETHPROXY) {
			Guard l(x_log);
			logerror << "Discarding incomplete response" << endl;
		}
		if (m_connected)
			readline();
	} else if (m_connected) {
		Guard l(x_log);
		logerror << "Read response failed: " + ec.message() << endl;
	}
}

void EthStratumClient::processExtranonce(std::string& enonce)
{
	m_extraNonceHexSize = enonce.length();

	{
		Guard l(x_log);
		logwarn << "Extranonce set to " << EthYellow << enonce << EthReset << endl;
	}

	enonce.resize(16, '0');
	m_extraNonce = h64(enonce);
}

void EthStratumClient::processReponse(Json::Value& responseObject)
{
	if (g_logJson) {
		Guard l(x_log);
		loginfo << "JSON RX" << endl << responseObject << endl;
	}
	Json::Value error = responseObject.get("error", {});
	if (error.isArray()) {
		Guard l(x_log);
		logerror << error.get(1, "Unknown error").asString() << endl;
	}
	std::stringstream json;
	std::ostream os(&m_requestBuffer);
	Json::Value params;
	int id = responseObject.get("id", Json::Value::null).asInt();
	switch (id) {
	case 1:
		if (m_connection.Version() == EthStratumClient::ETHEREUMSTRATUM) {
			m_nextWorkDifficulty = 1;
			params = responseObject.get("result", Json::Value::null);
			if (params.isArray()) {
				std::string enonce = params.get((Json::Value::ArrayIndex)1, "").asString();
				processExtranonce(enonce);
			}

			json << "{\"id\": 2, \"method\": \"mining.extranonce.subscribe\", \"params\": []}\n";
		}
		if (m_connection.Version() != EthStratumClient::ETHPROXY) {
			{
				Guard l(x_log);
				loginfo << "Subscribed to stratum server" << endl;
			}
			json << "{\"id\": 3, \"method\": \"mining.authorize\", \"params\": [\"" << m_connection.User() << "\",\"" <<
			     m_connection.Pass() << "\"]}\n";
		} else {
			m_authorized = true;
			json << "{\"id\": 5, \"method\": \"eth_getWork\", \"params\": []}\n"; // not strictly required but it does speed up initialization
		}
		os << json.str();

		async_write_with_response(m_requestBuffer);
		if (g_logJson)
			logJson(json.str());

		break;
	case 2:
		// nothing to do...
		break;
	case 3:
		m_authorized = responseObject.get("result", Json::Value::null).asBool();
		if (!m_authorized) {
			{
				Guard l(x_log);
				logerror << "Worker not authorized:" + m_connection.User() << endl;
			}
			throw runtime_error("Not authorized");
		}
		{
			Guard l(x_log);
			loginfo << "Authorized worker " + m_connection.User() << endl;
		}
		break;
	case 4: {
		m_responsetimer.cancel();
		m_response_pending = false;
		if (responseObject.get("result", false).asBool()) {
			if (m_onSolutionAccepted)
				m_onSolutionAccepted(m_stale);
		} else {
			if (m_onSolutionRejected)
				m_onSolutionRejected(m_stale);
		}
	}
	break;
	default:
		string method, workattr;
		unsigned index;
		if (m_connection.Version() != EthStratumClient::ETHPROXY) {
			method = responseObject.get("method", "").asString();
			workattr = "params";
			index = 1;
		} else {
			method = "mining.notify";
			workattr = "result";
			index = 0;
		}

		if (method == "mining.notify") {
			params = responseObject.get(workattr.c_str(), Json::Value::null);
			if (params.isArray()) {
				string job = params.get((Json::Value::ArrayIndex)0, "").asString();
				if (m_response_pending)
					m_stale = true;
				if (m_connection.Version() == EthStratumClient::ETHEREUMSTRATUM) {
					string sSeedHash = params.get((Json::Value::ArrayIndex)1, "").asString();
					string sHeaderHash = params.get((Json::Value::ArrayIndex)2, "").asString();

					if (sHeaderHash != "" && sSeedHash != "") {
						reset_work_timeout();

						m_current.header = h256(sHeaderHash);
						m_current.seed = h256(sSeedHash);
						m_current.boundary = h256();
						diffToTarget((uint32_t*)m_current.boundary.data(), m_nextWorkDifficulty);
						m_current.startNonce = ethash_swap_u64(*((uint64_t*)m_extraNonce.data()));
						m_current.exSizeBits = m_extraNonceHexSize * 4;
						m_current.job_len = job.size();
						if (m_connection.Version() == EthStratumClient::ETHEREUMSTRATUM)
							job.resize(64, '0');
						m_current.job = h256(job);

						if (m_onWorkReceived)
							m_onWorkReceived(m_current);
					}
				} else {
					string sHeaderHash = params.get((Json::Value::ArrayIndex)index++, "").asString();
					string sSeedHash = params.get((Json::Value::ArrayIndex)index++, "").asString();
					string sShareTarget = params.get((Json::Value::ArrayIndex)index++, "").asString();

					// coinmine.pl fix
					int l = sShareTarget.length();
					if (l < 66)
						sShareTarget = "0x" + string(66 - l, '0') + sShareTarget.substr(2);


					if (sHeaderHash != "" && sSeedHash != "" && sShareTarget != "") {
						h256 headerHash = h256(sHeaderHash);

						if (headerHash != m_current.header) {
							reset_work_timeout();

							m_current.header = h256(sHeaderHash);
							m_current.seed = h256(sSeedHash);
							m_current.boundary = h256(sShareTarget);
							m_current.job = h256(job);

							if (m_onWorkReceived)
								m_onWorkReceived(m_current);
						}
					}
				}
			}
		} else if (method == "mining.set_difficulty" && m_connection.Version() == EthStratumClient::ETHEREUMSTRATUM) {
			params = responseObject.get("params", Json::Value::null);
			if (params.isArray()) {
				m_nextWorkDifficulty = params.get((Json::Value::ArrayIndex)0, 1).asDouble();
				if (m_nextWorkDifficulty <= 0.0001) m_nextWorkDifficulty = 0.0001;
				{
					Guard l(x_log);
					logwarn << "Difficulty set to "  << EthYellow << m_nextWorkDifficulty << EthReset << endl;
				}
			}
		} else if (method == "mining.set_extranonce" && m_connection.Version() == EthStratumClient::ETHEREUMSTRATUM) {
			params = responseObject.get("params", Json::Value::null);
			if (params.isArray()) {
				std::string enonce = params.get((Json::Value::ArrayIndex)0, "").asString();
				processExtranonce(enonce);
			}
		} else if (method == "client.get_version") {
			json << "{\"error\": null, \"id\" : " << id << ", \"result\" : \"" << miner_get_buildinfo()->project_version <<
			     "\"}\n";
			os << json.str();

			async_write_with_response(m_requestBuffer);
			if (g_logJson)
				logJson(json.str());
		}
		break;
	}

}

void EthStratumClient::stop_timeout_handler(const boost::system::error_code& ec)
{
	if (!ec) {
		{
			Guard l(x_log);
			logerror << "Stopping as requested.\n";
		}
		exit(0);
	}
}

void EthStratumClient::work_timeout_handler(const boost::system::error_code& ec)
{
	if (!ec) {
		{
			Guard l(x_log);
			logerror << "No new work received in " << g_worktimeout << " seconds.\n";
		}
		throw runtime_error("Work timeout");
	}
}

void EthStratumClient::response_timeout_handler(const boost::system::error_code& ec)
{
	if (!ec) {
		{
			Guard l(x_log);
			logerror << "No no response received in 2 seconds." << endl;
		}
		throw runtime_error("Response timeout");
	}
}

void EthStratumClient::hr_timeout_handler(const boost::system::error_code& ec)
{
	if (!ec) {
		stringstream ss;
		ss << "0x" << hex << m_rate;
		// There is no stratum method to submit the hashrate so we use the rpc variant.
		string json = "{\"id\": 6, \"jsonrpc\":\"2.0\", \"method\": \"eth_submitHashrate\", \"params\": [\"" +
		              ss.str() + "\",\"0x" + m_submit_hashrate_id + "\"]}\n";
		std::ostream os(&m_hrBuffer);
		os << json;
		if (m_connection.SecLevel() != SecureLevel::NONE)
			async_write(*m_securesocket, m_hrBuffer,
			            boost::bind(&EthStratumClient::handleHashrateResponse, this, boost::asio::placeholders::error));
		else
			async_write(*m_socket, m_hrBuffer,
			            boost::bind(&EthStratumClient::handleHashrateResponse, this, boost::asio::placeholders::error));
		if (g_logJson)
			logJson(json);
	}
}

void EthStratumClient::submitHashrate(uint64_t rate)
{
	// Called by the pool manager thread.
	m_rate = rate;
	m_hrtimer.cancel();
	m_hrtimer.expires_from_now(boost::posix_time::milliseconds(10));
	m_hrtimer.async_wait(boost::bind(&EthStratumClient::hr_timeout_handler, this, boost::asio::placeholders::error));
}

void EthStratumClient::submitSolution(Solution solution)
{

	string nonceHex = toHex(solution.nonce);
	string json;

	m_responsetimer.cancel();

	switch (m_connection.Version()) {
	case EthStratumClient::STRATUM:
		json = "{\"id\": 4, \"method\": \"mining.submit\", \"params\": [\"" +
		       m_connection.User() + "\",\"" + solution.work.job.hex() + "\",\"0x" +
		       nonceHex + "\",\"0x" + solution.work.header.hex() + "\",\"0x" +
		       solution.mixHash.hex() + "\"]}\n";
		break;
	case EthStratumClient::ETHPROXY:
		json = "{\"id\": 4, \"worker\":\"" +
		       m_worker + "\", \"method\": \"eth_submitWork\", \"params\": [\"0x" +
		       nonceHex + "\",\"0x" + solution.work.header.hex() + "\",\"0x" +
		       solution.mixHash.hex() + "\"]}\n";
		break;
	case EthStratumClient::ETHEREUMSTRATUM:
		json = "{\"id\": 4, \"method\": \"mining.submit\", \"params\": [\"" +
		       m_connection.User() + "\",\"" + solution.work.job.hex().substr(0, solution.work.job_len) + "\",\"" +
		       nonceHex.substr(m_extraNonceHexSize, 16 - m_extraNonceHexSize) + "\"]}\n";
		break;
	}

	auto buf = new boost::asio::streambuf;
	m_submitBuffers.push(buf);
	std::ostream os(buf);
	os << json;
	m_stale = solution.stale;

	if (m_connection.SecLevel() != SecureLevel::NONE)
		async_write(*m_securesocket, *buf,
		            boost::bind(&EthStratumClient::handleSubmitResponse, this, boost::asio::placeholders::error));
	else
		async_write(*m_socket, *buf,
		            boost::bind(&EthStratumClient::handleSubmitResponse, this, boost::asio::placeholders::error));

	if (g_logJson)
		logJson(json);

	m_response_pending = true;
	m_responsetimer.expires_from_now(boost::posix_time::seconds(2));
	m_responsetimer.async_wait(boost::bind(&EthStratumClient::response_timeout_handler, this,
	                                       boost::asio::placeholders::error));
}

