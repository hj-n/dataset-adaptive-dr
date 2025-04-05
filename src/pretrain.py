class DatasetAdaptiveDR:
	def __init__(
		self, 
		dataset_path: str, ## path to the directory containing the datasets
		result_path: str,  ## path to the directory that will store the results
		intermediate_path: str, ## path to the directory that will store intermediate results
		metric: str, ## currently supports: "mnc", "pds", "pds+mnc", and... 
	):
		pass