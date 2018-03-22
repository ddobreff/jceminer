# ethminer

[![Join the chat at https://gitter.im/miscellaneousbits/miner](https://badges.gitter.im/miscellaneousbits/miner.svg)](https://gitter.im/miscellaneousbits/miner?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg)](https://github.com/RichardLitt/standard-readme)

> Ethereum miner with OpenCL, CUDA and stratum support. Cloned from ethrereum-mining/ethminer.

This is adapted for personal use. I make no claims of viability, and likely won't respond to bug reports or requests. Should
someone go to the trouble of pushing a decent change, I'll certainly take a look.

If you're going to try to use this, the release branches (1.0.0, 1.0.1, etc...) should be stable.

## Usage

For a full list of available command options, please run

```
Options:

  -h [ --help ]             produce help message.
                            
  -l [ --list ]             List devices.
                            
  -v [ --version ]          list version.
                            
  -f [ --file ] arg         Read parameters from file.
                            
  -r [ --retries ] arg (=3) Connection retries.
                            
  --email arg               Stratum email.
                            
  --timeout arg (=180)      Work timeout.
                            
  --hash                    Report hashrate to pool.
                            
  --intvl arg (=15)         statistics display interval.
                            
  --level arg (=0)          Metrics collection level. 0 - HR only, 1 - + fan & 
                            temp, 2 - + power.
                            
  -p [ --pool ] arg         URL takes the form:
                            scheme://[user[:password]@]hostname:port
                            
                            unsecured schemes: ethproxy+tcp nicehash+tcp stratum+tcp
                            secured with any TLS: ethproxy+tls nicehash+tls stratum+tls
                            secured with TLS 1.2: ethproxy+tls12 nicehash+tls12 stratum+tls12
                            
                            Example:
                            stratum+tls://0x0123456...7890190123.miner1@ethermine.org:5555
                            
                            Stratum versions:
                            stratum  - ethpool, ethermine, coinotron, mph, nanopool
                            eth-proxy - dwarfpool, f2pool, nanopool (required for hashrate reporting to work with nanopool)
                            nicehash - Ethereum Stratum
                            
  --dag arg (=0)            DAG load mode. 0 - parallel, 1 - sequential, 2 - single.
                            
  --switch                  Log job switch time.
                            
  --json                    Log formatted json messaging.
                            
  -G [ --cl ]               Opencl mode.
                            
  -U [ --cu ]               Cuda mode.
                            
  -X [ --mix ]              Mixed opencl and cuda mode. Use OpenCL + CUDA in a system with mixed AMD/Nvidia cards. May require setting --cl-plat 1 or 2.
                            
  --eval                    Enable software result evaluation. Use if you GPUs generate too many invalid shares.
                            
  --api arg (=0)            API port number. 0 - disable, < 0 - read-only.
                            
  --cl-plat arg (=0)        Opencl platform.
                            
  --cl-devs arg             Opencl device list.
                            
  --cl-kern arg (=0)        Opencl kernel. 0 - Opencl, 1 - binary.
                            
  --cu-grid arg (=8192)     Cuda grid size.
                            
  --cu-blk arg (=128)       Cuda block size.
                            
  --cu-devs arg             Cuda device list.
                            
  --cu-para arg (=4)        Cuda parallel hashes.
                            
  --cu-sch arg (=4)         Cuda schedule mode. 0 - auto, 1 - spin, 2 - yield, 4 - sync
                            
  --cu-strm arg (=2)        Cuda streams
                            
  --stop arg (=0)           Stop after minutes. 0 - never stop.

```

### Building from source

In addition to the CUDA dev tools, you will require the following packages installed to build the miner: git cmake mesa-common-dev libidn11-dev libgoogle-perftools-dev


   ```sh
   sudo apt install git cmake mesa-common-dev libidn11-dev libgoogle-perftools-dev
   ```

This project uses [CMake] and [Hunter] package manager.

1. Make sure git submodules are up to date

   ```sh
   git submodule update --init --recursive
   ```

2. Create a build directory.

   ```sh
   mkdir build; cd build
   ```

3. Configure the project with CMake. Check out additional
   [configuration options](#cmake-configuration-options).

   ```sh
   cmake ..
   ```

4. Build the project using make.

   ```sh
   make
   ```

5. Optional, install the built executable.

   ```sh
   sudo make install
   ```

### CMake configuration options

Pass these options to CMake configuration command, e.g.

```sh
cmake .. -DETHASHCUDA=ON -DETHASHCL=OFF
```

- `-DETHASHCL=ON` - enable OpenCL mining, `ON` by default,
- `-DETHASHCUDA=OFF` - enable CUDA mining, `ON` by default.
- `-DAPICORE=ON` - enable Api, `OFF` by default.


## License

__Regrettably this must be licensed under the [GNU General Public License, Version 3](LICENSE.md).__

