class TickerData:
    def __init__(self,load_yf: bool=False,
                 repo_url='https://github.com/laughingbud/capstone'):
        self.repo_url = repo_url
        # self.notebook_dir = os.getcwd()
        # self.notebook_dir = Path().absolute()
        # print(self.notebook_dir)
        if load_yf:
            # URL of the Yahoo Finance commodities page
            tickers = []
            descriptions = []
            for cat in ['commodities','currencies']:
                url = f'https://finance.yahoo.com/markets/{cat}/'
            # url = 'https://finance.yahoo.com/markets/commodities/'
            # https://finance.yahoo.com/markets/currencies/

                # Request the page content
                response = requests.get(url)
                soup = BeautifulSoup(response.content, 'html.parser')

                yf_attribute_dict = {'ticker':'symbol yf-ravs5v',
                                    'description':'yf-ravs5v longName'}
                for key,value in yf_attribute_dict.items():
                    long_text = soup.find_all('span', attrs={'class':value})
                    for val in long_text:
                        if key =='ticker':
                            tickers.append(str(val)[str(val).find(value)+len(value)+2:str(val).find("</span>")])
                        elif key =='description':
                            descriptions.append(str(val)[str(val).find(value)+len(value)+9:str(val).find('">')].replace("amp;", ""))
                            # print(str(val)[str(val).find(value)+len(value)+9:str(val).find('">')].replace("amp;", ""))

            # print(descriptions)
            config_df = pd.DataFrame({'Ticker':tickers,'Description':descriptions})
            print(config_df)
            config_df['Ticker'] = config_df['Ticker'].astype(str)
            config_df = config_df.set_index('Ticker') # Set 'Ticker' column as index
            config_df.loc[config_df.Ticker.str.contains('=F'),'Category'] = 'Futures'
            config_df.loc[config_df.Ticker.str.contains('=X'),'Category'] = 'Forward'
            config_df.loc[config_df.Category=='Futures','Sub_category'] = 'Commodity'
            config_df.iloc[:4,3] = 'Equity'
            config_df.iloc[4:8,3] = 'Bond'
            config_df.loc[config_df.Category=='Forward','Sub_category'] = 'Currency'
            self.config = config_df
            self.tickers = tickers
            self.data = {}

    def get_tickers(self,category=None,sub_category=None):
        config = self.config
        if (category == None)&(sub_category==None):
            return self.tickers
        elif (category == None)&(sub_category!=None):
            return config.loc[config.Sub_category==sub_category,'Ticker'].tolist()
        elif (category != None)&(sub_category==None):
            return config.loc[config.Category==category,'Ticker'].tolist()
        else:
            return config.loc[(config.Category==category) & (config.Sub_category==sub_category),'Ticker'].tolist()

    def get_config(self):
        return self.config

    def is_directory_empty(self, directory):
        return not any(os.scandir(directory))

    def filter_large_csv(self,file_path, filter_value, filter_column=None,
                         chunksize=10000,date_filter=None):
        filtered_data = pd.DataFrame()

        for chunk in pd.read_csv(file_path, chunksize=chunksize):
            if filter_column is None:
                filter_column = chunk.columns[0]  # Default to the first column if not specified
            filtered_chunk = chunk[chunk[filter_column] == filter_value]
            filtered_data = pd.concat([filtered_data, filtered_chunk])

        # filtered_data['Datetime'] = pd.to_datetime(filtered_data['Datetime'], errors='coerce')
        # display(filtered_data.dtypes)
        # #Fix the data type of the Datetime column
        # filtered_data['Datetime'] = pd.to_datetime(filtered_data['Datetime'], errors='coerce')
        # #Now localize the Datetime column
        # filtered_data['Datetime'] = filtered_data['Datetime'].dt.tz_localize('UTC').dt.tz_convert('America/New_York')
        # filtered_data['Datetime'] = pd.to_datetime(filtered_data['Datetime']).tz_localize('UTC').tz_convert('America/New_York')

        if date_filter != None:
            filtered_data['Datetime'] = pd.to_datetime(filtered_data['Datetime']).dt.tz_localize('America/New_York')
            filtered_data = filtered_data.loc[filtered_data.Datetime<date_filter]

        # filtered_data = filtered_data.reset_index(inplace=True,drop=True)

        # filtered_data['Datetime'] = pd.to_datetime(filtered_data['Datetime'], errors='coerce')
        # filtered_data = filtered_data.set_index('Datetime')
        # filtered_data.index = pd.DatetimeIndex(filtered_data.index)
        # display(filtered_data.shape)
        # filtered_data['hour'] = filtered_data.index.hour
        # filtered_data['hour'] = pd.DatetimeIndex( filtered_data.index).hour
        filtered_data=filtered_data.drop(columns=['Ticker'])
        return filtered_data


    def load_git(self,repo_url,file_name,filter_name,date_filter):
        repo_name = re.search(r'/([^/]+)\.git$', repo_url).group(1)
        print(f'Repo name: {repo_name}')
        # Get the directory of the current file
        # current_file_directory = os.path.dirname(os.path.realpath(__file__))
        # current_file_directory=os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        # Change the current working directory to the directory of the current file
        if os.path.exists(f'/content/{repo_name}'):
            os.chdir(f'/content/{repo_name}')

        print(f'Current working directory: {os.getcwd()}')
        if os.path.exists(f'/content/{repo_name}'):
            if self.is_directory_empty(f'/content/{repo_name}'):
                subprocess.run(["git", "clone", repo_url, repo_name])
                print("Empty directory. Cloning.")
            else:
                print(f".....Directory '{repo_name}' exists and is not empty.")
        else:
            subprocess.run(["git", "clone", repo_url, repo_name])
            print("Directory doesn't exist. Cloning.")
        # !git clone https://github.com/laughingbud/conquer.git
        # !git clone {repo_url}

        os.chdir(f'/content/{repo_name}')

        # Open the ZIP file
        with zipfile.ZipFile(file_name, 'r') as zip_ref:
        # Extract all contents to the current directory
            zip_ref.extractall()

        with zipfile.ZipFile(file_name, 'r') as zip_ref:
            print(zip_ref.namelist())

        df = self.filter_large_csv(zip_ref.namelist()[0],
                                   filter_value=filter_name,filter_column=None,
                                   chunksize=10000,date_filter=date_filter)

        return df

#    def store_git(self,file_name):
#        !git config --global user.email "ditesh.verma@gmail.com"
#        !git config --global user.name "laughingbud"
#        # !got config --global user.password "wrongpassword"
#
#        username = 'laughingbud'
#        repo = 'conquer'
#        !git clone https://{token}@github.com/{username}/{repo}
#
#        all_data.to_csv(f'conquer/all_ticker_intraday_data_{all_data.index.get_level_values(1).unique()[0].strftime("%Y-%m-%d")}.csv')
#        !git status
#        !git add --all
#        !git commit -a -m "adding intraday data"
#        !git status
#        !git remote -v
#        !git commit -a -m "adding intraday data"
#        !git push origin main
#
#        # self.config.to_csv(file_name)
#        return None

    def download_data(self, tickers, start_date, interval='1h',
                      default_max_period=False):
        today = datetime.today()
        last_bday = (today - BDay(1)).strftime("%Y-%m-%d")

        if interval=='1h':
            # Maximum of last 730 calendar days can be queried for YF
            max_start_date = (today-timedelta(730)+ BDay(1)).strftime("%Y-%m-%d")

            date1_obj = datetime.strptime(start_date.strftime("%Y-%m-%d"),
                                                  '%Y-%m-%d')
            date2_obj = datetime.strptime(max_start_date, '%Y-%m-%d')
            if date1_obj < date2_obj:
                print("Given date is earlier than allowed by YF.")
            else:
                print("Given date is within the max period allowed by YF.")

            if default_max_period == True:
                adj_start_date = max_start_date
                print("Defaulting to Max allowed start date.")
            else:
                adj_start_date = start_date
                print("Defaulting to user defined start date.")

        ticker_data = pd.DataFrame()
        for ticker in tickers:
            ticker_obj = yf.Ticker(ticker)
            if interval=='1h':
                ticker_data = ticker_obj.history(
                    start=adj_start_date,interval=interval)
            else:
                ticker_data = ticker_obj.history(start=start_date.strftime("%Y-%m-%d"),
                                                 interval=interval)
            if ticker_data.empty:
                print(f"No data found for {ticker}")
                continue
            else:
                # ticker_data['rets'] = ticker_data['Close'].pct_change()
                ticker_data['rets'] = np.log(
                    ticker_data['Close'] / ticker_data['Close'].shift(1))
                # ticker_data['rets'] = (ticker_data['Close'] - ticker_data['Open'])/ticker_data['Open']

                if interval.find('h')==1:
                    ticker_data['hour'] = ticker_data.index.hour
                #     ticker_data['range'] = ticker_data['High'] - ticker_data['low']
                #     ticker_data['std'] = ticker_data['rets'].rolling(
                #         window=24).std()*np.sqrt(24)
                # del ticker_data['Volume']
                del ticker_data['Dividends']
                del ticker_data['Stock Splits']
                ticker_data = ticker_data.dropna()
                self.data[ticker] = ticker_data
        return None

    def load_hourly_data(self,tickers, history: bool=True):
        today = datetime.today()
        last_bday = (today - BDay(1)).strftime("%Y-%m-%d")
        self.download_data(tickers=tickers,start_date=(today - BDay(10330)),
              interval='1h',default_max_period=True)
        if history:
            for ticker in tickers:
                hist = self.load_git(repo_url='https://github.com/laughingbud/capstone.git',
                                    file_name='all_ticker_intraday_data_2022-09-02.zip',
                                    filter_name=ticker,date_filter=None)
                hist=hist.loc[pd.to_datetime(hist.Datetime)<self.data[ticker].index[0]].set_index('Datetime')
                hist['hour'] = pd.to_datetime(hist.index).hour

                self.data[ticker] = pd.concat([hist,self.data[ticker]],axis=0)
                self.data[ticker].index = pd.to_datetime(self.data[ticker].index, utc=True)
                self.data[ticker].index = self.data[ticker].index.tz_convert('America/New_York')
        return None

    def get_minutely_tickers(self,repo_dir='capstone/Data'):
        # Clone the GitHub repository
        repo_url = self.repo_url
        local_dir = f'/content/{repo_dir}'

        if not os.path.exists(local_dir):
            Repo.clone_from(repo_url,repo_dir.split('/')[0])
        os.chdir(local_dir)

        # List all CSV files in the repository
        csv_files = []
        for root, dirs, files in os.walk(local_dir):
            for file in files:
                # print(file)
                if file.endswith('.csv'):
                    csv_files.append(os.path.join(root, file))

        print(f"{len(csv_files)} CSV files found:", csv_files)

        def is_number(s):
            try:
              float(s)
              return True
            except ValueError:
              return False

        months_sc = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        months_uc = [month.upper() for month in months_sc]
        months_uc.extend(months_sc)
        tickers = []
        for file in csv_files:
            csv_path = os.path.join(local_dir, file)
            file_name = os.path.basename(csv_path)
            company_name = file_name.split('_')[0]
            if company_name not in tickers:
                if ".csv" in company_name:
                    short_name = company_name.split('.csv')[0]
                    if len(short_name) >5 and is_number(short_name[-2:]) and short_name[-5:-2] in months_uc:
                        company_name = short_name[:-5]
                    else:
                        continue
                else:
                    continue


                tickers.append(company_name)
        tickers = sorted(list(set(tickers)))
        print(f'{len(tickers)} tickers returned.')
        ticker_dict = {}
        for ticker in tickers:

            # ticker_dict[ticker] = {"files":[file for file in csv_files if ticker in file]
            #                        }
            ticker_dict[ticker] = {"files":[file for file in csv_files if file.split('/')[5].startswith(ticker) and "Missing" not in file.split('/')[4]]
                        }
            ticker_dict[ticker]['months'] = list(set([file.split('/')[4] for file in ticker_dict[ticker]['files'] if "Missing" not in file.split('/')[4]]))
            ticker_dict[ticker]['n_months'] = len(set(ticker_dict[ticker]['months']))
            ticker_dict[ticker]['n_files'] = len(set(ticker_dict[ticker]['files']))

        self.minutely_tickers = tickers
        self.minutely_ticker_dict = ticker_dict
        return tickers,ticker_dict

    def load_minutely_data(self,repo_dir='capstone/Data',ticker=None,filter=None):

        # Clone the GitHub repository
        repo_url = self.repo_url
        local_dir = f'/content/{repo_dir}'
        if not os.path.exists(local_dir):
            Repo.clone_from(repo_url,repo_dir.split('/')[0])
        os.chdir(local_dir)

        # List all CSV files in the repository
        csv_files = []
        if ticker != None:
            csv_files = self.minutely_ticker_dict[ticker]['files']
        else:
            for root, dirs, files in os.walk(local_dir):
                for file in files:
                    # print(file)
                    if file.endswith('.csv'):
                        csv_files.append(os.path.join(root, file))

        # print(f"{len(csv_files)} CSV files found:", csv_files)

        dataframes = []
        combined_df = pd.DataFrame()
        for file in csv_files:

            if filter == None or filter in file:
                csv_path = os.path.join(local_dir, file)
                print(f"Reading CSV file: {csv_path}")
                df = pd.DataFrame()
                for chunk in pd.read_csv(csv_path,chunksize=10000):
                    # chunk.columns = [col.strip() for col in chunk.columns]
                    # if not df.empty and chunk.columns != df.columns:
                    #     print(f"The CSV file: {csv_path} has column mismatch.")
                    #     print(f"Expected columns: {df.columns}")
                    #     print(f"Actual columns: {chunk.columns}")
                        # continue
                    df = pd.concat([df,chunk],axis=1)

                df.columns = [col.strip() for col in df.columns]
                new_columns = [col.replace('<', '').replace('>', '') for col in df.columns]
                df.columns = new_columns

                if len(dataframes)>0:
                    if not len(df.columns)==len(dataframes[0].columns):
                        print(f'Actual number of columns {len(df.columns)}')
                        print(f'Expected number of columns {len(dataframes[0].columns)}')
                        print(f"The CSV file: {csv_path} has column mismatch.")
                        print(f"Expected columns: {dataframes[0].columns}")
                        print(f"Actual columns: {df.columns}")
                        df = df.iloc[:,:len(dataframes[0].columns)]
                        print(f"Truncated to: {df.columns}")

                    df.columns = dataframes[0].columns

                df['date'] = df['date'].apply(lambda x: x.replace('-', '/'))
                # print(pd.to_datetime(df.date, format='mixed').dt.month.unique())
                try:
                    df['date'] = pd.to_datetime(df.date)
                except ValueError:
                    df['date'] = pd.to_datetime(df.date, format='%d/%m/%Y')
                # print(f'Dataframe shape: {df.shape}')
                # print(df.date.dt.month.unique())
                # df.date = pd.to_datetime(df.date) if len(pd.to_datetime(df.date).dt.date.unique())>1 else pd.to_datetime(df.date,format='%d/%m/%Y')
                dataframes.append(df)
            else:
                continue

        # Check if any dataframes were found
        if not dataframes:
            print(f"No CSV files found matching the filter '{filter}' in '{repo_dir}'.")
            return pd.DataFrame()  # Return an empty DataFrame if no dataframes are found

        # Combine all dataframes into one
        combined_df = pd.concat(dataframes, ignore_index=True)

        combined_df['datetime'] = pd.to_datetime(
            combined_df['date'].astype(str) + ' ' + combined_df['time'])
        combined_df = combined_df.set_index('datetime').sort_index()
        combined_df = combined_df.drop(columns=['date','time'])
        combined_df['return'] = np.log(combined_df['close'] / combined_df['close'].shift(1))
        combined_df['return']=combined_df['return'].fillna(0)

        return combined_df

        # return None

def convert_to_datetime(row):
    # try:
    #     return pd.to_datetime(row['date']).dt.strftime('%d/%m/%Y')
    # except ValueError:
    #     if '-' in row['date']:
    #         try:
    #             return pd.to_datetime(row['date'].replace('-', '/')).dt.strftime('%d/%m/%Y')
    #             # row['date'] = row['date'].replace('-', '/')
    #             # row['date'] = pd.to_datetime(row['date']).dt.strftime('%d/%m/%Y')
    #         except ValueError:
    #             print(f"Could not convert date and time for row: {row}")

    try:
      return pd.to_datetime(pd.to_datetime(row['date']).dt.strftime('%d/%m/%Y') + ' ' + row['time'])
    except ValueError:
      # Handle cases where the date format is different
      if isinstance(row['date'], str) and '/' in row['date']:
        try:
          return pd.to_datetime(row['date'].replace('/', '-') + ' ' + row['time'])
        except ValueError:
          print(f"Could not convert date and time for row: {row}")
          return pd.NaT
