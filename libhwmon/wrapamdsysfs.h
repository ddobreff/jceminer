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

#pragma once

typedef struct {
	int sysfs_gpucount;
	int opencl_gpucount;
	int* card_sysfs_device_id;  /* map cardidx to filesystem card idx */
	int* sysfs_hwmon_id;        /* filesystem card idx to filesystem hwmon idx */
	int* sysfs_opencl_device_id;          /* map ADL dev to OPENCL dev */
	int* opencl_sysfs_device_id;          /* map OPENCL dev to ADL dev */
} wrap_amdsysfs_handle;

wrap_amdsysfs_handle* wrap_amdsysfs_create();
int wrap_amdsysfs_destroy(wrap_amdsysfs_handle* sysfsh);

int wrap_amdsysfs_get_gpucount(wrap_amdsysfs_handle* sysfsh, int* gpucount);

int wrap_amdsysfs_get_gpu_pci_id(wrap_amdsysfs_handle* sysfsh, int index, char* idbuf, int bufsize);

int wrap_amdsysfs_get_tempC(wrap_amdsysfs_handle* sysfsh, int index, unsigned int* tempC);

int wrap_amdsysfs_get_fanpcnt(wrap_amdsysfs_handle* sysfsh, int index, unsigned int* fanpcnt);

int wrap_amdsysfs_get_power_usage(wrap_amdsysfs_handle* sysfsh, int index, unsigned int* milliwatts);
