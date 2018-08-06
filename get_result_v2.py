import os
import time
import pandas as pd
base = '/home/bing/cu_learn/im2col/src/'
path = ['MIVReRAM', 'TSVReRAM','ReRAM']
#path = ['alexnet_TSV/']
#path = ['alexnet_ReRAM','alexnet_2DDRAM']
#path = ['ReRAM']
result = 'result.csv'


def get_timepowerFile(current_path):
	timeFile={}
	powerFile={}
	timepowerFile={}
	os.chdir(current_path)
	for files in os.listdir(current_path):
		if files.find("FF")!=-1:
			timeFile[time.ctime(os.path.getmtime(files))]=files
		elif files.endswith("log"):
			powerFile[time.ctime(os.path.getmtime(files))]=files
	for item, values in timeFile.items():
		powerfile=powerFile.get(item)
		if powerfile is not None:
			timepowerFile[values]=powerfile
	return timepowerFile

def write_file(rsfile,string):
	fout = open(rsfile,'w')
	fout.write(headline+'\n')
	fout.close()

def get_result(path):
	for subpath in path:
		os.chdir(base+subpath)
		current_path = base+subpath
		print(current_path)
		timepowerFile=get_timepowerFile(current_path)
		headline='name\ttot_cycle\ttot_energy\ttot_edp\ttot_dram_energy\ttot_mc_energy\n'	
		for times, powers in timepowerFile.items():
			timelog2, powerlog2, powerlog3 =[],[],[]
			cycle, power = 0, 0
			fs = open(times,'r+')
			pfile = open(powers,'r+')
			for line in fs.readlines():
				a=line.split()
				if a!=[]: 
					if a[0].endswith("tot_sim_cycle"):
						cycle = float(a[-1])
					elif a[0]=="gpu_sim_cycle":
						timelog2.append(float(a[-1]))
			for line in pfile.readlines():
				a=line.split()
				if a!=[]: 
					if a[0].endswith("tot_avg_power"):
						power = float(a[-1])
					elif a[0].startswith("gpu_avg_DRAMP"):
						powerlog2.append(float(a[-1]))
					elif a[0].startswith("gpu_avg_MCP"):
						powerlog3.append(float(a[-1]))
			print(cycle,power)
			tot_energy = cycle*power*1.41
			tot_cycle = cycle*1.41
			tot_edp = cycle**2 * power*1.41
			"""
			print("gpu_simcycle:",timelog2)
			print("avg_DRAMP:", powerlog2)
			print("avg_MCP", powerlog3)
			"""
			dram_energy,mc_energy=0,0
			for i in range(len(timelog2)):
				dram_energy += timelog2[i]*powerlog2[i]
				mc_energy +=timelog2[i]*powerlog3[i]
			print("dram_energy:",dram_energy, "mc_energy:", mc_energy)
			headline+=times+'\t'+str(tot_cycle)+'\t'+str(tot_energy)+'\t'+str(tot_edp)+'\t'+str(dram_energy)+'\t'+str(mc_energy)+'\n'	

		rsfile = subpath+result
		write_file(rsfile,headline)


def find_layers(layers,name):
	for i in range(len(layers)):
		layer = layers[i]
		if 'name' in layer.keys():
			if layer['name'] == name:
				return i
#def reoutput_kernel():
def reoutput_kernel(base):
	#base = '/home/bing/cu_learn/im2col/src/ReRAM'
	timepowerFile = get_timepowerFile(base)
	i = 0
	headline='name\tkernel_latency(ns)\tkernel_energy(nJ)\n'	
	for times, powers in timepowerFile.items():
		#print(times)
		kernel, timelog, powerlog =[],[],[]
		fs = open(times,'r+')
		pfile = open(powers,'r+')
		for line in fs.readlines():
			a=line.split()
			if a!=[]: 
				if a[0].endswith("kernel_name"):
					kernel.append(a[-1])
				elif a[0]=="gpu_sim_cycle":
					timelog.append(float(a[-1])*1.42)
		for line in pfile.readlines():
			a=line.split()
			if a!=[]: 
				if a[0].endswith("kernel_avg_power"):
					powerlog.append(float(a[-1]))
		"""
		for i in range(len(kernel)):
			print(kernel[i])
		"""
		conv_idx, fc_idx, pool_idx, relu_idx=0,0,0,0
		layers = []
		i=0
		while(i<len(kernel)):
			#print("process:",kernel[i])
			layer={}
			ff_layer={}
			kernel_energy = 0
			if kernel[i].find('im2col')!=-1:
				fctime = 0
				name=''
				# BP
				if kernel[i+1].find('conv')!=-1:
					ff_idx = find_layers(layers,'conv'+str(conv_idx))
					ff_layer = layers[ff_idx]
					for j in range(2):
						kernel_energy += timelog[i+j]*powerlog[i+j]
						fctime += timelog[i+j]	
						headline+="[{name:"+name+',G_latency:'+str(fctime)+',G_energy:'+str(kernel_energy)+','
						ff_layer['G_latency']=fctime
						ff_layer['G_energy']=kernel_energy
					stride=4	
				#FP
				else:	
					stride=3
				for j in range(stride):
					kernel_energy += timelog[i+j]*powerlog[i+j]
					fctime += timelog[i+j]	
				if stride == 4:
					headline+='BPlatency:'+str(fctime)+',BPenergy:'+str(kernel_energy)+'}],'
					ff_layer['BPlatency']=fctime 
					ff_layer['BPenergy']=kernel_energy
					conv_idx = conv_idx-1
					i=i+stride
					continue
				else:
					conv_idx = conv_idx+1
					name='conv'+str(conv_idx)
					headline+="[{name:"+name+',latency:'+str(fctime)+',energy:'+str(kernel_energy)+'}],'
					layer={"name":name,'latency':fctime,'energy':kernel_energy}
					i=i+stride
			elif kernel[i].find('fc')!=-1: 
				fctime = 0
				name=''
				if kernel[i+1].find('fc')!=-1:
					ff_idx = find_layers(layers,'fc'+str(fc_idx))
					ff_layer = layers[ff_idx]
					kernel_energy = timelog[i]*powerlog[i]
					fctime = timelog[i]	
					headline+="[{name:"+name+',G_latency:'+str(fctime)+',G_energy:'+str(kernel_energy)+','
					ff_layer['G_latency']=fctime
					ff_layer['G_energy']=kernel_energy
					stride=3	
				else:
					stride=2
				for j in range(stride):
					kernel_energy += timelog[i+j]*powerlog[i+j]
					fctime += timelog[i+j]	
				if stride ==3:
					headline+=',BPlatency:'+str(fctime)+',BPenergy:'+str(kernel_energy)+'}],'
					ff_layer['BPlatency']=fctime 
					ff_layer['BPenergy']=kernel_energy
					fc_idx = fc_idx-1
					i=i+stride
					continue
				else:
					fc_idx = fc_idx+1
					headline+="[{name:"+name+',latency:'+str(fctime)+',energy:'+str(kernel_energy)+'}],'
					name='fc'+str(fc_idx)
					layer={"name":name,'latency':fctime,'energy':kernel_energy}
					i=i+stride
			
			elif kernel[i].find('Pool')!=-1:
				kernel_energy = timelog[i]*powerlog[i]
				fctime = timelog[i]	
				if kernel[i].find("Backward")!=-1:
					ff_idx = find_layers(layers,'pool'+str(pool_idx))
					if ff_idx is not None:
						ff_layer = layers[ff_idx]
						ff_layer['BPlatency']=fctime 
						ff_layer['BPenergy']=kernel_energy
						pool_idx = pool_idx-1	
					i=i+1
					#print(ff_layer)
					continue
				else:
					kernel_energy = timelog[i]*powerlog[i]
					headline+="[{name:"+kernel[i]+',latency:'+str(timelog[i])+',energy:'+str(kernel_energy)+'}],'	
					pool_idx = pool_idx+1
					name='pool'+str(pool_idx)
					layer={"name":name,'latency':fctime,'energy':kernel_energy}
					i=i+1
			elif kernel[i].find('Re')!=-1:
				kernel_energy = timelog[i]*powerlog[i]
				fctime = timelog[i]	
				if kernel[i].find("Backward")!=-1:
					ff_idx = find_layers(layers,'relu'+str(relu_idx))
					if ff_idx is not None:
						ff_layer = layers[ff_idx]
						ff_layer['BPlatency']=fctime 
						ff_layer['BPenergy']=kernel_energy
						relu_idx = relu_idx-1	
					i=i+1
					#print(ff_layer)
					continue
				else:
					kernel_energy = timelog[i]*powerlog[i]
					headline+="[{name:"+kernel[i]+',latency:'+str(timelog[i])+',energy:'+str(kernel_energy)+'}],'	
					relu_idx = relu_idx+1
					name='relu'+str(relu_idx)
					layer={"name":name,'latency':fctime,'energy':kernel_energy}
					i=i+1
			#print(layer)
			layers.append(layer)
		print("{}={}".format(times,layers))
		#print(headline)
		"""
		rsfile = base+'kernel.csv'
		write_file(rsfile,layers)
		"""

for subpath in path:
	print(subpath)
	reoutput_kernel(base+subpath)

#reoutput_kernel()
