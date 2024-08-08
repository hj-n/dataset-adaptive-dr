import numpy as np 
import pandas as pd

pplusminus_result = pd.read_csv("./app/results/app2_pplusminus.csv")
pzero_result = pd.read_csv("./app/results/app2_prandom.csv")



average_result = {}
for metric in ["tnc_25", "mrre_25", "l_tnc_0", "srho_0", "pr_0"]:
	average_result[metric] = {}
	localplusglobalminus = pplusminus_result[pplusminus_result["pm"] == "plus"]
	localplusglobalminus = localplusglobalminus[pplusminus_result["rankingby"] == "mncpds"]
	# pass
	average_result[metric]["local+global-"] = localplusglobalminus[localplusglobalminus["metric"] == metric]["correlation"].mean()

	localplus = pplusminus_result[pplusminus_result["pm"] == "plus"]
	localplus = localplus[pplusminus_result["rankingby"] == "mnc"]

	average_result[metric]["local+"] = localplus[localplus["metric"] == metric]["correlation"].mean()

	globalminus = pplusminus_result[pplusminus_result["pm"] == "minus"]
	globalminus = globalminus[pplusminus_result["rankingby"] == "pds"]

	average_result[metric]["global-"] = globalminus[globalminus["metric"] == metric]["correlation"].mean()

	random = pzero_result[pzero_result["metric"] == metric]["correlation"].mean()

	average_result[metric]["random"] = random


	localminus = pplusminus_result[pplusminus_result["pm"] == "minus"]
	localminus = localminus[pplusminus_result["rankingby"] == "mnc"]

	average_result[metric]["local-"] = localminus[localminus["metric"] == metric]["correlation"].mean()


	globalplus = pplusminus_result[pplusminus_result["pm"] == "plus"]
	globalplus = globalplus[pplusminus_result["rankingby"] == "pds"]

	average_result[metric]["global+"] = globalplus[globalplus["metric"] == metric]["correlation"].mean()



	globalpluslocalminus = pplusminus_result[pplusminus_result["pm"] == "minus"]
	globalpluslocalminus = globalpluslocalminus[pplusminus_result["rankingby"] == "mncpds"]

	average_result[metric]["global+local-"] = globalpluslocalminus[globalpluslocalminus["metric"] == metric]["correlation"].mean()


metric_list = []
method_list = []
correlation_list = []

for metric in average_result:
	for method in average_result[metric]:
		
		metric_list.append(metric)
		method_list.append(method)
		correlation_list.append(average_result[metric][method])

df = pd.DataFrame({
	"metric": metric_list,
	"method": method_list,
	"correlation": correlation_list
})



df.to_csv("./app/results/app2_results.csv", index=False)



