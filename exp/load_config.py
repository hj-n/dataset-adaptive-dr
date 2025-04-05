
def load_config(key):
		"""
		Load the configuration from the config.yaml file.
		
		Args:
				key (str): The key to retrieve from the config file.
				
		Returns:
				dict: The value associated with the key in the config file.
		"""
		import yaml
		with open("config.yaml", "r") as f:
				config = yaml.safe_load(f)
		return config[key]