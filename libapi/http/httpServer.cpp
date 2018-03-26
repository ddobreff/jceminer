#include <chrono>
#include <thread>
#include <mongoose/mongoose.h>
#include "httpServer.h"
#include "libdevcore/Log.h"
#include "libdevcore/Common.h"
#include "miner-buildinfo.h"

using namespace dev;
using namespace eth;

httpServer http_server;

void httpServer::getstat1(stringstream& ss)
{
	using namespace std::chrono;
	auto info = miner_get_buildinfo();
	WorkingProgress p = m_farm->miningProgress();
	SolutionStats s = m_farm->getSolutionStats();
	string l = m_farm->farmLaunchedFormatted();
	ss << "Version: " << info->project_version << "\r\n";
	ss << p << s << ' ' << l << "\r\n";
}

static void static_getstat1(stringstream& ss)
{
	http_server.getstat1(ss);
}

void ev_handler(struct mg_connection* c, int ev, void* p)
{

	if (ev == MG_EV_HTTP_REQUEST) {
		struct http_message* hm = (struct http_message*) p;
		char uri[32];
		unsigned uriLen = hm->uri.len;
		if (uriLen >= sizeof(uri) - 1)
			uriLen = sizeof(uri) - 1;
		memcpy(uri, hm->uri.p, uriLen);
		uri[uriLen] = 0;
		if (::strcmp(uri, "/getstat1"))
			mg_http_send_error(c, 404, nullptr);
		else {
			stringstream content;
			static_getstat1(content);
			mg_send_head(c, 200, (int)content.str().length(), "Content-Type: text/plain");
			mg_printf(c, "%.*s", (int)content.str().length(), content.str().c_str());
		}
	}
}

void httpServer::run(unsigned short port, dev::eth::Farm* farm)
{
	m_farm = farm;
	m_port = to_string(port);
	new thread(bind(&httpServer::run_thread, this));
}

void httpServer::run_thread()
{
	struct mg_mgr mgr;
	struct mg_connection* c;

	mg_mgr_init(&mgr, NULL);
	loginfo << "Starting web server on port " << m_port << '\n';
	c = mg_bind(&mgr, m_port.c_str(), ev_handler);
	if (c == NULL) {
		logerror << "Failed to create listener\n";
		return;
	}

	// Set up HTTP server parameters
	mg_set_protocol_http_websocket(c);

	for (;;)
		mg_mgr_poll(&mgr, 1000);
}

httpServer::httpServer()
{
}

httpServer::~httpServer()
{
}


