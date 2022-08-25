import hub
import streamlit as st

@hub.compute
def filter_labels(sample_in, labels_list):

    return sample_in.labels.data()['text'][0] in labels_list

class ActiveLoop():
    def __init__(self,dataset:str):
        self.dataset_endpoint= "hub://activeloop"
        self.dataset=dataset

    def get_train(self):
        train_ds = self.dataset + "-train"
        url = "/".join([self.dataset_endpoint,train_ds])
        return hub.load(url)

    def get_test(self):
        test_ds = self.dataset + "-test"
        url = "/".join([self.dataset_endpoint,test_ds])
        return hub.load(url)

if __name__ == '__main__':

    AL = ActiveLoop("fashion-mnist")

    ds_train = AL.get_train()
    ds_test= AL.get_test()

    st.write("Active Loop Train Dataset")
    st.write(ds_train)
    st.write("Active Loop Test Dataset")
    st.write(ds_train)

    labels_list = [0,1] # Desired labels for filtering

    ds_view = ds_train.filter(filter_labels(labels_list), scheduler = 'threaded', num_workers = 0)
    st.write("Filter DS")
    st.write(ds_view)