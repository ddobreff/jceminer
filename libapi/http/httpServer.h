#pragma once

#include <thread>
#include <libethcore/Farm.h>

class httpServer
{
public:
	httpServer(unsigned short port, dev::eth::Farm& farm);
	~httpServer();
private:
	static void serve(class httpServer*);
	unsigned short m_port;
	dev::eth::Farm& m_farm;
	class httpServer* m_server;
};

