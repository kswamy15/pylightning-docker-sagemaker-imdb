import os, re, wget, tarfile, time

class ImdbData():
    # download the original IMDB dataset in tgz compressed format
    # os.system('wget https://s3.amazonaws.com/fast-ai-nlp/imdb.tgz')
    
    # Untar the tgz file - It will create an imdb directory with a subdirectory for train and test data.  The train and test directory will have pos and neg subdirectories.
    # The pos directory will contain all the positive reviews text and the neg directory will contain all the negative reviews text.  There are 25,000 text files each in the train
    # and test data.  Of the 25,000 text files, 12,500 are positive review text files and 12,500 are negative review text files.  It is a balanced data set.  
    def __init__(self) -> None:
        data_source_url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
        self.data_file_path='/opt/ml/input/aclImdb_v1.tar.gz'
        self.data_folder='/opt/ml/input/aclImdb'
        if not self.check_if_file_exists(self.data_file_path):
            print('Start of data download')
            wget.download(url=data_source_url, out='/opt/ml/input')
            print('Download complete')
        else:
            print('Data file already exists. Not downloading again!')

        if not self.check_if_dir_exists(self.data_folder):
            startTime = time.time()
            my_tar = tarfile.open(self.data_file_path)
            print('Extracting all the files now...')
            my_tar.extractall('/opt/ml/input')
            my_tar.close()
            print('Done!') 
            total_time=time.time()-startTime
            print('Time Taken for extracting all files : ',total_time/60,'minutes')
        else:
            print('Data folder exists. Won\'t copy again!')    

    def check_if_file_exists(self, file):
        '''
        Checks if 'file' exists
        '''
        try:
            tarfh = tarfile.open(file)
            return True
        except FileNotFoundError:
            #print('Please make sure file: ' + file + ' is present before continuing')
            return False

    def check_if_dir_exists(self, directory):
        '''
        Checks if 'directory' exists
        '''
        return(os.path.isdir(directory))

    """
    Load file into memory
    """
    def load_file(self, filename):
        """
        Open the file as read only
        """
        file = open(filename, 'r')
        """
        Read all text
        """
        text = file.read()
        """
        Close the file
        """
        file.close()
        return text.strip()    

    ## Remove the HTML tags like <br> in the text
    def preprocess_reviews(self, review):
        #REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\d+)")
        REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
        NO_SPACE = ""
        SPACE = " "
        #reviews = [REPLACE_NO_SPACE.sub(NO_SPACE, line.lower()) for line in reviews]
        review = REPLACE_WITH_SPACE.sub(SPACE, review)
    
        return review
 
    def get_data(self):
        """
        Reading train test directory
        """
        reviews_dict={'train':[],'test':[]}
        for split in ['train','test']:
            directory = os.path.join(self.data_folder,split)
            for sentiment in ['pos', 'neg']:
                data_folder=os.path.join(directory, sentiment)
                print('Data Folder : ',data_folder)
                for root, dirs, files in os.walk(data_folder):
                    for fname in files:
                        if fname.endswith(".txt"):
                            file_name_with_full_path=os.path.join(root, fname)
                            review=self.load_file(file_name_with_full_path)
                            clean_review=self.preprocess_reviews(review)
                            if split == 'train':
                                reviews_dict['train'].append(clean_review)
                            else:
                                reviews_dict['test'].append(clean_review)
        return reviews_dict
# This is the main method, to be run when train.py is invoked
if __name__ =='__main__':
    imdbdata1 = ImdbData()
    reviews = imdbdata1.get_data()
    print(reviews['train'][0])