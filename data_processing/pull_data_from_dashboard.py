import datetime
import requests
from requests.auth import HTTPBasicAuth
import json
import numpy as np
from pprint import pprint
import copy
import time
import os

################################ Preparing for Queries ################################
bigtimer = datetime.datetime.now()
REALMS = ['rubicon-fmap', 'rubicon-fbmp', 'nbcuni-superstore', 'jumpshot-jsc', 'rubicon-fbmq', 'rally-health-integration', 'prudential-nj-exp2', 'rubicon-fmaq']
# the name of the realms configured in pepperdata
for realm in REALMS:
	# base URL for queries
	base_url = 'https://beta-dashboard.pepperdata.com/{0}/api/m'.format(realm)

	# number of days to query
	NUM_DAYS = 7

	END_TIME = 1505242800000  
	# Filename to output data
	FILENAME_APP = '/home/ubuntu/data_processing/cnn/data/' + realm + '_' + str(END_TIME) + '_' + str(NUM_DAYS) + '_days_app_data.json'

	# Relevant time and batch size for jobs
	# END_TIME = datetime.datetime.now()-datetime.timedelta(days=1)

	# the max number of jobs that will be returned for the time period
	max_series = 100000

	# apiusername and password
	api_key_username = ''
	api_key_password = ''

	# time between samples in milliseconds
	sample_millis = 1

	# timezone
	time_zone_hour_offset = -7

	# output format
	output_format = 'JSON'

	# configures the pepperdata parameters
	pepperdata_params = {
	  'sample': sample_millis,
	  'tzo': time_zone_hour_offset,
	  'format': output_format,
	  'ms': max_series,
	  'sample':1
	}

	# Required header for queries
	headers = {'Accept': 'application/json'}

	app_metrics = ['vmsram', 'tasks', 't_rscthnetno', 't_rscthhfsrb', 'c_ucpupct']
	# app_metrics = ['tasks', 'rssram', 'c_ucpupct', 't_rscthhfsrbps', 't_rscthhfswbpps', 't_rscthnetrbpps', 't_rscthnetwbpps']

	queue_metrics = ['tasks', 'rm_used_containers', 'rm_ask_containers']

	################################ Job Querying ################################

	# Prepare for looping over number of days to decrease load on API
	job_info = {}
	# Structure: {job_id: job_name: name, job_user: user, job_queue: queue, job_start: start, job_finish: finish, tasks: timeseriesfortasks, ... for all metrics}
	pepperdata_params['j'] = '*'

	ONE_DAY_IN_UNIX_TIME = 86400000

	for metric in app_metrics:
	    pepperdata_params['m']=metric 

	    end = END_TIME
	    all_series = []

	    # Request day chunks of the time range and add all the "all_series" to all_series
	    for counter in range(0,NUM_DAYS):
		# Set start and end times
		start = end - ONE_DAY_IN_UNIX_TIME

		pepperdata_params['s'] = start
		pepperdata_params['e'] = end

		# Make the http request for one day and time the response
		print("Making request for jobs...")
		timer = datetime.datetime.now()
		boolean = False
		while not boolean:
		    try:
			response = requests.get(base_url, pepperdata_params, auth=HTTPBasicAuth(api_key_username, api_key_password),
						headers=headers)
			boolean = True
		    except:
			time.sleep(60)
		timer = datetime.datetime.now()-timer
		print ("query number " + str(counter+1) + " for jobs took this long: " + str(timer))

		# parse json response and dump useful information into all_series
		json_response = response.json()
		try:
		    json_data = json_response['data']
		except: 
		    print "ERROR, this was the response: "
		    print json_response
		    continue
		all_series += json_data['allSeries']
		end -= ONE_DAY_IN_UNIX_TIME

	    # Parse every job's information contained in the all_series and pull out the information needed for job_info (see above)
	    length = float(len(all_series))
	    for i, job in enumerate(all_series): 
		job_id = job['fullJobId']
		# Add the job_id to the keys in job_info if it doesnt exist and give it all the basic information of the job
		if job_id not in job_info.keys():
		    job_info[job_id] = {}
		    try:
			job_info[job_id]['job_name'] = job['name']['seriesNameMap']['JOB_NAME']
			job_info[job_id]['job_user'] = job['name']['seriesNameMap']['USER']
			job_info[job_id]['job_queue'] = job['name']['seriesNameMap']['QUEUE']
			job_info[job_id]['job_start'] = job['name']['seriesNameMap']['JOB_START']
			job_info[job_id]['job_finish'] = job['name']['seriesNameMap']['JOB_FINISH']
		    except:
			continue
		# Add the metrics time series that's being queried for to the job_info dictionary 
		job_info[job_id][metric] = job['dataPoints']
		percentage = int((i/length)*100.0)
		if percentage%10 == 0:
		    print "%d percent through the jobs info."%percentage


	# Dump the data in a file
	with open(FILENAME_APP, 'w') as outfile:
	    json.dump(job_info, outfile, indent=4)

	bigtimer = datetime.datetime.now() - bigtimer

print "This all took %s much time"%bigtimer



