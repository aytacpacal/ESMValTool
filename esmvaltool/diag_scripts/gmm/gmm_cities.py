# operating system manipulations (e.g. path constructions)
import os
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn import metrics
import math
from os import getcwd, makedirs
from os.path import exists
from scipy import special as sps
from datetime import datetime
from cdo import *
import yaml
import traceback
import multiprocessing
import functools

# to manipulate iris cubes
import iris
import matplotlib.pyplot as plt

# import internal esmvaltool modules here
from esmvaltool.diag_scripts.shared import group_metadata, run_diagnostic
from esmvalcore.preprocessor import area_statistics



# era5data = '/mnt/lustre02/work/bd1083/b309178/era5cli/out/era5_2m_temperature_1980-2010_daymax_masked.nc'

# city_path = "/mnt/lustre01/pf/b/b309178/ExtremeEvents/cities2018.csv"
# cities = pd.read_csv(city_path,
                     # sep=',',
                     # header=0,
                     # dtype={'lon': np.float64, 'lat': np.float64},
                     # encoding='cp1258')
# cities = cities.round({'lat': 4, 'lon': 4})
# cities = cities.loc[(cities['2018'] >= 20000000)]
# print(cities)
# title_font = {'fontsize': 12, 'fontweight': 'normal'}
# label_font = {'fontsize': 12}

def _plot_time_series(cfg, cube, dataset):
    """
    Example of personal diagnostic plotting function.

    Arguments:
        cfg - nested dictionary of metadata
        cube - the cube to plot
        dataset - name of the dataset to plot

    Returns:
        string; makes some time-series plots

    Note: this function is private; remove the '_'
    so you can make it public.
    """
    # custom local paths for e.g. plots are supported -
    # here is an example
    # root_dir = '/group_workspaces/jasmin2/cmip6_prep/'  # edit as per need
    # out_path = 'esmvaltool_users/valeriu/'   # edit as per need
    # local_path = os.path.join(root_dir, out_path)
    # but one can use the already defined esmvaltool output paths
    local_path = cfg['plot_dir']

    # do the plotting dance
    plt.plot(cube.data, label=dataset)
    plt.xlabel('Time (months)')
    plt.ylabel('Area average')
    plt.title('Time series at (ground level - first level)')
    plt.tight_layout()
    plt.grid()
    plt.legend()
    png_name = 'Time_series_' + dataset + '.png'
    plt.savefig(os.path.join(local_path, png_name))
    plt.close()

    # no need to brag :)
    return 'I made some plots!'


def dict_print(d):
    for k, v in d.items():
        if isinstance(v, dict):
            print("\n\n")
            print("KEY - ", k)
            dict_print(v)
        else:
            print(k, " : ", v)
    
    
def chi2test(obs, exp, bin=1):
    print(int(min([min(obs), min(exp)])))
    print(int(max([max(obs), max(exp)])))
    sumchi = 0
    for i in range(int(min([min(obs), min(exp)])), int(max([max(obs), max(exp)]))+1, bin):
        olen = len(obs[(obs < i+1) & (obs > i-1)])
        elen = len(exp[(obs < i+1) & (exp > i-1)])
        if elen != 0:
            sumchi += (olen-elen)**2/elen
    return sumchi


def SelBest(arr:list, X:int)->list:
    '''
    returns the set of X configurations with shorter distance
    '''
    dx=np.argsort(arr)[:X]
    return arr[dx]


def gmm_silohuette(dataset, c, m, p):
    n_clusters = np.arange(2, 11)
    sils = []
    sils_err = []
    iterations = 20
    print("test for best silohuette")
    for n in n_clusters:
        tmp_sil = []
        for _ in range(iterations):
            gmm = GaussianMixture(n, n_init=2).fit(X=np.expand_dims(dataset, 1))
            labels = gmm.predict(X=np.expand_dims(dataset, 1))
            holder = np.expand_dims(dataset,1)
            sil = metrics.silhouette_score(np.expand_dims(dataset, 1), labels, metric='euclidean')
            tmp_sil.append(sil)
        val = np.mean(SelBest(np.array(tmp_sil), int(iterations / 5)))
        err = np.std(tmp_sil)
        sils.append(val)
        sils_err.append(err)
    plt.figure()
    plt.errorbar(n_clusters, sils, yerr=sils_err)
    plt.title("Silhouette Scores", fontsize=20)
    plt.xticks(n_clusters)
    plt.xlabel("N. of clusters")
    plt.ylabel("Score")
    plt.savefig(
        output_directory + '/' + c + '_GMM_silhoutte_comparison_' + m + '_' + p + '.png')
    plt.close()
    return 1


def gmm_select(dataset, c, m, p):
    lowest_bic = np.infty
    best_n = 0
    best_init=0
    bic = []
    bic_plot = []
    scores = {}
    n_components_range = range(1, 10)
    for n_components in n_components_range:
        inits = {}
        temp_bic = []
        for n_init in range(1, 11):
            # Fit a Gaussian mixture with EM
            gmm = GaussianMixture(n_components=n_components,
                                  covariance_type='full',
                                  n_init=n_init)
            gmm.fit(X=np.expand_dims(dataset, 1))
            inits[n_init] = gmm.score(X=np.expand_dims(dataset, 1))
            bic.append(gmm.bic(X=np.expand_dims(dataset, 1)))
            temp_bic.append(gmm.bic(X=np.expand_dims(dataset, 1)))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm
                best_n = n_components
                best_init = n_init
        bic_plot.append(sum(temp_bic)/len(temp_bic))
        scores[n_components] = inits
    #bic = np.array(bic)

    bars = []
    # Plot the BIC scores
    plt.figure()
    xpos = np.array(n_components_range)
    barplot = plt.bar(xpos, bic_plot)
    plt.xticks(n_components_range)
    print(min(bic_plot), max(bic_plot))
    plt.ylim(bottom=min(bic_plot)-200, top=max(bic_plot)+200)
    plt.title('BIC score per model')
    #xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 + \
           #.2 * np.floor(bic.argmin() / len(n_components_range))
    #plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
    plt.yticks(np.arange(min(bic_plot), max(bic_plot), 200))
    plt.title('Selected GMM for' + c + ' ' + m + ' ' + p)
    #plt.subplots_adjust(hspace=.35, bottom=.02)
    plt.savefig(
        output_directory + '/' + c + '_GMM_comparison_' + m + '_' + p + '.png')
    plt.close()
    return [best_gmm, best_n, scores, best_init]


def data_extract(lon, lat, ncfile):
    remap = "remapnn"
    print(ncfile + ' start data extract using ' + remap)
    options = ' -' + remap + ',lon=' + str(lon) + '_lat=' + str(lat)
    loc_data = pd.DataFrame([re.split('\s+', i) for i in cdo.outputtab("name,lon,lat,year,month,day,value" + options, input=ncfile)][1:],
                            columns=['name', 'lon', 'lat', 'year', 'month', 'day','value'])
    loc_data[['lon', 'lat', 'value']] = loc_data[['lon', 'lat', 'value']].astype(float)
    loc_data[['year', 'month', 'day']] = loc_data[['year', 'month', 'day']].astype(int)
    loc_data = loc_data.round({'lat': 4, 'lon': 4})
    if 'era' in ncfile:
        loc_data['value'] -= 273.15
        loc_data = loc_data.loc[(loc_data['year'] >= 1980) & (loc_data['year'] <= 2009)]
    loc_data = loc_data.merge(cities, on=['lon', 'lat'], how='inner')
    return loc_data


def round_down(num, divisor):
    if num < 0:
        return num - (10 - (-1 * num % divisor))
    else:
        return num - (num % divisor)



def round_up(num, divisor):
    if num % divisor <= 5:
        return num + (5 - num % divisor)
    else:
        return num + (10 - num % divisor)    


#def histogram_creator():

def main(city, cfg):
    print("Starting " + city[0])
    print(multiprocessing.current_process())
    # assemble the data dictionary keyed by dataset name
    # this makes use of the handy group_metadata function that
    # orders the data by 'dataset'; the resulting dictionary is
    # keyed on datasets e.g. dict = {'MPI-ESM-LR': [var1, var2...]}
    # where var1, var2 are dicts holding all needed information per variable
    my_files_dict = group_metadata(cfg['input_data'].values(), 'dataset')
    global outputPath
    outputPath = cfg['plot_dir']
    
    # READ ERA5 DATA 
    era5data = '/mnt/lustre02/work/bd1083/b309178/era5cli/out/era5_2m_temperature_1980-2010_daymax_masked.nc'


    title_font = {'fontsize': 12, 'fontweight': 'normal'}
    label_font = {'fontsize': 12}
    # READ MODEL NAMES FROM NESTED DICTIONARY FOR FILES
    model_list = [*my_files_dict]
    for key, values in my_files_dict.items():
        if key == 'ERA-Interim':
            continue
        holder = [value['filename'] for value in values]
   
    # iterate over key(dataset) and values(list of vars)
    
    for model in model_list:
        print(model)
        if model == 'ERA-Interim':
            continue
        match_dict = {}
        # SEARCH FOR DIFFERENT PERIODS OF THE SAME MODEL
        matches = [match for match in holder if model in match]
        matches.extend([era5data])
        print('Assign models to dict ' + city[0])
        # ASSIGN EACH FILE TO DICT KEYS. LEN(MATCHES) must be 4, i.e. era5, historical, ssp370 and ssp585.
        if len(matches) == 4:
            for item in matches:
                if 'historical' in item:
                    match_dict['past'] = item
                elif 'era5' in item:
                    match_dict['era'] = item
                elif '_ssp370_' in item:
                    match_dict['future370'] = item
                elif '_ssp585_' in item:
                    match_dict['future585'] = item
                else:
                    print(item)
                    print('Error, what should I do with it?')
                    break
        else:
            print(matches)
            print('Matches are not equal to 4!!')
            break
            # SELECT KEYS FOR OUTPUT DIRECTORY
        print(*cfg["input_data"])
        output_folder_keys = [cfg["input_data"][match_dict['past']]['project'],
                              cfg["input_data"][match_dict['past']]['dataset'],
                              cfg["input_data"][match_dict['past']]['short_name'], 
                              str(cfg["input_data"][match_dict['past']]['start_year']) + '-' + 
                              str(cfg["input_data"][match_dict['past']]['end_year']),
                              str(cfg["input_data"][match_dict['future585']]['start_year']) + '-' +
                              str(cfg["input_data"][match_dict['future585']]['end_year']),
                              ]
        global output_directory
        output_directory = outputPath + dt_string + '/' + '_'.join(str(v) for v in output_folder_keys)
        print('Output directory: ' + output_directory)
        # CREATE DIRECTORIES IF NOT EXIST
        if not exists(output_directory):
            makedirs(output_directory)
                
        # import cdo outputs, append future, past and era of the model
        # set lat and lon columns float precission

        key = 'Maximum Temperature'
        cdo_out = {}
        # loop over city_names in dataset
        print('start cities loop\n')
        for city_name in [item for item in city if not pd.isna(item)]:
    #        try:
            city_lon = cities.loc[(cities['city']==city_name)]['lon'].values[0]
            city_lat = cities.loc[(cities['city']==city_name)]['lat'].values[0]
            print(city_name + ' start data extract')
            cdo_out['past'] = data_extract(lon=city_lon, lat=city_lat, ncfile=match_dict['past'])
            cdo_out['era'] = data_extract(lon=city_lon, lat=city_lat, ncfile=match_dict['era'])
            cdo_out['future370'] = data_extract(lon=city_lon, lat=city_lat, ncfile=match_dict['future370'])
            cdo_out['future585'] = data_extract(lon=city_lon, lat=city_lat, ncfile=match_dict['future585'])
            print(city_name + ' end data extract')

            histMin = round_down(min(cdo_out['past']['value']), 10)
            histMax = round_up(max(cdo_out['future585']['value']), 10)

            excelColumns = ['city', 'variable',
                            'mu_hot_past', 'sigma_hot_past', 'mu_cold_past', 'sigma_cold_past', 'hotPeriodDays_past',
                            'peak_diff_past',
                            'mu_hot_future', 'sigma_hot_future', 'mu_cold_future', 'sigma_cold_future',
                            'hotPeriodDays_future', 'peak_diff_future', 'change_peak_dif',
                            '1-year event', '10-year event', '30-year event', '50-year event', '100-year event']
            excel_holder = pd.DataFrame(columns=excelColumns)

            print(city_name, model, key, 'assign histograms')

            hist_past = cdo_out['past']['value']
            hist_future = cdo_out['future585']['value']
            hist_era = cdo_out['era']['value']

            print('start GMMs')
            # GMM ERA
            gmm_era = gmm_select(hist_era, c=city_name, m=model, p='era')
            #gmm_era_silohouttte = gmm_silohuette(hist_era, c=city_name, m=model, p='era')
            gmm_era_fit = gmm_era[0].fit(X=np.expand_dims(hist_era, 1))
            print('Era data for ' + city_name + ' follows ' + str(gmm_era[1]) + ' component GMM')
            gmm_era_fit_sample = gmm_era[0].sample(hist_era.size)

            # gmm_past FIT
            gmm_past = gmm_select(hist_past, c=city_name, m=model, p='past')
            gmm_past_fit = gmm_past[0].fit(X=np.expand_dims(hist_past, 1))
            print('History data for ' + city_name + ' follows ' + str(gmm_past[1]) + ' component GMM')
            gmm_past_fit_sample = gmm_past[0].sample(hist_past.size)

            # gmm_future FIT
            gmm_future = gmm_select(hist_future, c=city_name, m=model, p='future')
            gmm_future_fit = gmm_future[0].fit(X=np.expand_dims(hist_future, 1))
            print('Future data for ' + city_name + ' follows ' + str(gmm_future[1]) + ' component GMM')
            gmm_future_fit_sample = gmm_future[0].sample(hist_future.size)

            #expectation = gmm_future_fit_sample[0].reshape(hist_future.size)
            #chi = spstat.chisquare(hist_future, expectation)
            #mychi = chi2test(hist_future, expectation, 1)

            # CREATE GAUSSIAN FOR PAST. USE np.exp TO REPRODUCE THE PROBABILITIES FROM LOGS
            gmm_past_x = np.linspace(int(min(hist_past)), int(max(hist_past)), hist_past.size)
            gmm_past_y = np.exp(gmm_past_fit.score_samples(gmm_past_x.reshape(-1, 1)))

            gmm_era_x = np.linspace(int(min(hist_era)), int(max(hist_era)), hist_era.size)
            gmm_era_y = np.exp(gmm_era_fit.score_samples(gmm_era_x.reshape(-1, 1)))

            # CREATE GAUSSIAN FROM PARAMETERS. USE np.exp TO REPRODUCE THE PROBABILITIES FROM LOGS
            gmm_future_x = np.linspace(int(min(hist_future)), int(max(hist_future)), hist_future.size)
            gmm_future_y = np.exp(gmm_future_fit.score_samples(gmm_future_x.reshape(-1, 1)))

            # PLOT HISTOGRAM AND GAUSSIAN FIT
            fig = plt.figure(figsize=(6.1, 5), dpi=150)
            ax = fig.add_subplot(111)

            fig.subplots_adjust(left=0.125, right=0.98, bottom=0.1, top=0.95)

            # PLOT HISTOGRAM
            n, bins, patches = ax.hist([hist_past, hist_future, hist_era],
                                       bins=np.arange(histMin, histMax),
                                       alpha=0.15, color=['teal', 'firebrick', 'black'],
                                       density=1,
                                       label=[
                                           str(cdo_out['past']['year'].min()) + '-' +
                                           str(cdo_out['past']['year'].max()) + ' Reference',
                                           str(cdo_out['future585']['year'].min()) + '-' +
                                           str(cdo_out['future585']['year'].max()) + ' SSP585',
                                           str(cdo_out['era']['year'].min()) + '-' +
                                           str(cdo_out['era']['year'].max()) + ' ERA5'
                                              ]
                                       )

            # PLOT FUTURE AND PAST FIT
            ax.plot(gmm_past_x, gmm_past_y,
                    color="blue",
                    lw=2,
                    label=str(cdo_out['past']['year'].min()) + '-' + str(cdo_out['past']['year'].max()) + ' GMM')
            ax.plot(gmm_era_x, gmm_era_y,
                     color="grey",
                     lw=2,
                     ls='--',
                     label=str(cdo_out['era']['year'].min()) + '-' + str(cdo_out['era']['year'].max()) + ' GMM')
            ax.plot(gmm_future_x, gmm_future_y,
                    color="red",
                    lw=2,
                    label=str(cdo_out['future585']['year'].min()) + '-' + str(cdo_out['future585']['year'].max()) + ' GMM')

            # Annotate diagram
            ax.set_ylabel("Probability density", fontdict=label_font)
            ax.set_xlabel(key + " ($^\circ$C)", fontdict=label_font)

            # DEFINE MU AND SIGMA
            mu_past_cold = min(gmm_past_fit.means_)
            mu_past_hot = max(gmm_past_fit.means_)
            mu_past_cold_index = np.argmin(gmm_past_fit.means_)
            mu_past_hot_index = np.argmax(gmm_past_fit.means_)
            sigma_past_cold = math.sqrt(gmm_past_fit.covariances_[mu_past_cold_index])
            sigma_past_hot = math.sqrt(gmm_past_fit.covariances_[mu_past_hot_index])

            mu_future_cold = min(gmm_future_fit.means_)
            mu_future_hot = max(gmm_future_fit.means_)
            mu_future_cold_index = np.argmin(gmm_future_fit.means_)
            mu_future_hot_index = np.argmax(gmm_future_fit.means_)
            sigma_future_cold = math.sqrt(gmm_future_fit.covariances_[mu_future_cold_index])
            sigma_future_hot = math.sqrt(gmm_future_fit.covariances_[mu_future_hot_index])

            back_one_sigma = 50 + 100 * sps.erf(1 / math.sqrt(2)) / 2

            # hotterData = cityDataHolder.loc[(cityDataHolder['name'] == key) &
            #                                 (cityDataHolder['year'] >= pastStart) &
            #                                 (cityDataHolder['year'] <= pastEnd)]

            hotter_days_past = cdo_out['past'].loc[(cdo_out['past']['value'] >= mu_past_hot[0] - sigma_past_hot)]

            hotter_days_future = cdo_out['future585'].loc[
                (cdo_out['future585']['value'] >= mu_future_hot[0] - sigma_future_hot)]

            hot_period_number_days_past = (hotter_days_past.shape[0] * (100 / back_one_sigma)) / 30
            hot_period_number_days_future = (hotter_days_future.shape[0] * (100 / back_one_sigma)) / 30

            peak_diff_past = mu_past_hot[0] - mu_past_cold[0]
            peak_diff_future = mu_future_hot[0] - mu_future_cold[0]
            change_peak_dif = peak_diff_future - peak_diff_past

            return_period_holder = [city_name, key,
                                    mu_past_hot[0], sigma_past_hot, mu_past_cold[0], sigma_past_cold,
                                    hot_period_number_days_past, peak_diff_past,
                                    mu_future_hot[0], sigma_future_hot, mu_future_cold[0], sigma_future_cold,
                                    hot_period_number_days_future, peak_diff_future, change_peak_dif]

            for multiplier in [1, 10, 30, 50, 100]:
                # expected frequency of n-year events in the past Equation 3.20
                past_return_period = multiplier * hot_period_number_days_past

                # the expected frequency outside range Equation 3.21
                sigma_range = math.sqrt(2) * sps.erfinv(1 - (1 / past_return_period))

                # temperature limit of range Equation 3.22
                tau = mu_past_hot[0] + sigma_range * sigma_past_hot

                # calculation of x in mu-x*sigma for future Equation 3.23
                future_sigma_range = (tau - mu_future_hot[0]) / sigma_future_hot

                # changed value of n-year event Equation 3.24
                future_return_period = 1 / (1 - sps.erf(future_sigma_range / math.sqrt(2)))

                # Equation 3.25
                future_return_period_year = future_return_period / hot_period_number_days_future

                return_period_holder.extend([future_return_period_year])


            df = pd.DataFrame([return_period_holder], columns=excelColumns)
            excel_holder = excel_holder.append(df)

            # PLOT THE GRAPH
            ax.set_title(city_name.capitalize() + ' ' + model + ' | '
                         + str(cdo_out['past']['year'].min()) + '-' + str(cdo_out['past']['year'].max())
                         + ' vs. ' + str(cdo_out['future585']['year'].min()) + '-' + str(cdo_out['future585']['year'].max()),
                         fontdict=title_font)
            plt.xlim(min(gmm_past_x) - 5, max(gmm_future_x) + 5)

            # Draw legend
            plt.legend(fontsize=10, loc=0)

            # SAVE FIG
            # plt.show()
            plt.savefig(
                 output_directory + '/' + city_name + '_' + model + '.png')
            plt.close()
            print("Excel saved as \n" + output_directory + '/' + city_name + '_' + model + '_yearevents.xlsx')
            excel_holder.to_excel(output_directory + '/' + city_name + '_' + model + '_yearevents.xlsx', index=False)
            print('\nDONE')

            
    # that's it, we're done!
    return 'I am done with my first ESMValTool diagnostic!'

    
def run_my_diagnostic(cfg):
    global cdo
    global dt_string
    cdo = Cdo()
    dt_string = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(dt_string)
    city_path = "/mnt/lustre01/pf/b/b309178/ExtremeEvents/cities2018.csv"
    global cities
    cities = pd.read_csv(city_path,
                     sep=',',
                     header=0,
                     dtype={'lon': np.float64, 'lat': np.float64},
                     encoding='cp1258')
    cities = cities.round({'lat': 4, 'lon': 4})
    cities = cities.loc[(cities['2018'] >= 25000000)]
    print("Cities 25000000")    

    pool = multiprocessing.Pool(processes = 10)
    #pool.map(test, [[item] for item in cities['city'] if not pd.isna(item)])
    pool.map(functools.partial(main, cfg=cfg), [[item] for item in cities['city'] if not pd.isna(item)])
    

    
if __name__ == '__main__':
    # always use run_diagnostic() to get the config (the preprocessor
    # nested dictionary holding all the needed information)
    with run_diagnostic() as config:
        # list here the functions that need to run
        run_my_diagnostic(config)
