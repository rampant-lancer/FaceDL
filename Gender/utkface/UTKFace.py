import os
import tarfile
import numpy as np 
import cv2
import requests

class UTKFace(object):

    def __init__(self):
        '''
            To initialize the self.req_file variable with the zipped file and also to initialize
            self.data_dir where extracted data is stored
        '''
        self.req_file = 'downloaded/UTKFace.tar.gz'
        self.data_dir = 'downloaded/UTKFace'
        self.base_dir = 'downloaded'
        self.file_id = '0BxYys69jI14kYVM3aVhKS1VhRUk'
        
        curr_dir = os.listdir()
        if not 'downloaded' in curr_dir:
            os.mkdir('downloaded')

    
    def download_file_from_google_drive(self, id, destination):
        '''
            The implementation of this function and it's helper function is taken from : 
            https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url

            It takes the file id of the shared google drive file and download it using requests package
        '''
        URL = "https://docs.google.com/uc?export=download"

        session = requests.Session()

        response = session.get(URL, params = { 'id' : id }, stream = True)
        token = self.get_confirm_token(response)

        if token:
            params = { 'id' : id, 'confirm' : token }
            response = session.get(URL, params = params, stream = True)

        self.save_response_content(response, destination)

    
    def get_confirm_token(self, response):
        '''This is the helper function of the above download_file_from_google_drive function'''
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        
        return None

    
    def save_response_content(self, response, destination):
        '''This is the helper function of the above download_file_from_google_drive function'''
        CHUNK_SIZE = 32768
        print('Processing...')

        with open(destination, 'wb') as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)



    def get_data(self):
        '''
            This function reads the self.data_dir i.e "UTKFace" directory
            and convert those images into numpy array and returns them as X and Y

            @return1: X: Image data in numpy array format with dtype as np.uint8
                         X.shape = [None, 200, 200, 3]
            @return2: Y: Class label in numpy array format with dtype as np.int32
                         Y.shape = [None, ]
            @param1: countOfImages: Number of images that are to be converted to numpy array
                     positive integer to represent number of images NB: only for debugging or 
                     -1 to represent all images

            @param2: mode: The class label which would be extracted from the data
                     0 or 'age' for age
                     1 or 'gender' for gender
                     2 or 'race' for race

        '''

        X = []
        Y = []

        all_files = os.listdir(self.data_dir)
        _range = len(all_files)
        
        if self.countOfImages == -1 :
            sample_size = _range
        else:
            sample_size = self.countOfImages

        # Making sure that the selection is random.
        random_indices = np.random.choice(range(_range), sample_size, replace=False)
        for indx in random_indices:
            fpath = self.data_dir + '/' + all_files[indx]
            img = cv2.imread(fpath)
            img = cv2.resize(img, shape)
            label = int(all_files[indx].split('_')[self.mode])

            X.append(img)
            Y.append(label)
        
        return np.array(X), np.array(Y)
        



    def load_data(self, countOfImages, mode, shape):
        '''
            This is the main function of the class this function will extract images and labels
            from the directory specified.
        '''
        all_files = os.listdir(self.base_dir)
        self.countOfImages = countOfImages
        self.mode = mode
        self.shape = shape

        if not self.req_file.split('/')[1] in all_files:
            '''
                This is to deal with the case when the required file is not
                present in the current downloaded directory
            '''
            self.download_file_from_google_drive(self.file_id, self.req_file)

        if not self.data_dir.split('/')[1] in all_files:
            tar = tarfile.open(self.req_file, "r:gz")
            tar.extractall(self.base_dir)
            tar.close()

        return self.get_data()
