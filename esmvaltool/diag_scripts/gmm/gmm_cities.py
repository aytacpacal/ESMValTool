# operating system manipulations (e.g. path constructions)
import os

# to manipulate iris cubes
import iris
import matplotlib.pyplot as plt

# import internal esmvaltool modules here
from esmvaltool.diag_scripts.shared import group_metadata, run_diagnostic
from esmvalcore.preprocessor import area_statistics


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
            print(k)
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
    print(ncfile + ' start data extract')
    options = ' -remapdis,lon=' + str(lon) + '_lat=' + str(lat)
    loc_data = pd.DataFrame([re.split('\s+', i) for i in cdo.outputtab("name,lon,lat,year,month,day,value"+options,
                                                                       input=ncfile)][1:],
                            columns=['name', 'lon', 'lat', 'year', 'month', 'day', 'value'])
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

def run_my_diagnostic(cfg):
    """
    Simple example of a diagnostic.

    This is a basic (and rather esotherical) diagnostic that firstly
    loads the needed model data as iris cubes, performs a difference between
    values at ground level and first vertical level, then squares the
    result.

    Before plotting, we grab the squared result (not all operations on cubes)
    and apply an area average on it. This is a useful example of how to use
    standard esmvalcore.preprocessor functionality within a diagnostic, and
    especially after a certain (custom) diagnostic has been run and the user
    needs to perform an operation that is already part of the preprocessor
    standard library of functions.

    The user will implement their own (custom) diagnostics, but this
    example shows that once the preprocessor has finished a whole lot of
    user-specific metrics can be computed as part of the diagnostic,
    and then plotted in various manners.

    Arguments:
        cfg - nested dictionary of metadata

    Returns:
        string; runs the user diagnostic

    """
    # assemble the data dictionary keyed by dataset name
    # this makes use of the handy group_metadata function that
    # orders the data by 'dataset'; the resulting dictionary is
    # keyed on datasets e.g. dict = {'MPI-ESM-LR': [var1, var2...]}
    # where var1, var2 are dicts holding all needed information per variable
    my_files_dict = group_metadata(cfg['input_data'].values(), 'dataset')
    print(my_files_dict)
    print("\n\n\n\n PRINT START")
    dict_print(cfg)
    print("PRINT END \n\n\n\n ")
    # iterate over key(dataset) and values(list of vars)
    for key, value in my_files_dict.items():
        # load the cube from data files only
        # using a single variable here so just grab the first (and only)
        # list element
        cube = iris.load_cube(value[0]['filename'])
        print(cube)
        # the first data analysis bit: simple cube difference:
        # perform a difference between ground and first levels
        #diff_cube = cube[:, 0, :, :] - cube[:, 1, :, :]
        # square the difference'd cube just for fun
        #squared_cube = diff_cube ** 2.

        # the second data analysis bit (slightly more advanced):
        # compute an area average over the squared cube
        # to apply the area average use a preprocessor function
        # rather than writing your own function
        #area_avg_cube = area_statistics(squared_cube, 'mean')

        # finalize your analysis by plotting a time series of the
        # diffed, squared and area averaged cube; call the plot function:
        # _plot_time_series(cfg, area_avg_cube, key)

    # that's it, we're done!
    return 'I am done with my first ESMValTool diagnostic!'


if __name__ == '__main__':
    # always use run_diagnostic() to get the config (the preprocessor
    # nested dictionary holding all the needed information)
    with run_diagnostic() as config:
        # list here the functions that need to run
        run_my_diagnostic(config)
