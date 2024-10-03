import numpy as np
import pandas as pd
import random
import joblib
from sklearn.pipeline import make_pipeline
##############################################
##############
ncell=500000
num_train=15000
num_test=9000
num_valid=6000
rand_seed=42
#######
training_out_file="../tmp/synthetic_samples/training_input_auto/"
valid_out_file="../tmp/synthetic_samples/valid_input_auto/"
testing_out_file="../tmp/synthetic_samples/testing_input_auto/"
#####
###########################
    
def get_neg_df(neg_df,rand_seed,guid_counts,ncell_pos):
    fcs_count=0
    bool_var=False
    if ncell_pos<10000:
        ncells = 10000
    else:
        ncells= 10000
    while not bool_var:
        random.seed(rand_seed)
        rand_fcs = random.choices(guid_counts,k=fcs_count+1)
        cell_count=0
        guids=[]
        for item in rand_fcs:
            cell_count=cell_count+item[1]
            guids.append(item[0])
        return_df = neg_df.loc[neg_df['case'].isin(guids)]
        if len(return_df)>ncells:
            bool_var=True
        else:       
            fcs_count+=1
                
    if len(return_df)>=ncells:
        if len(return_df)<=20000:
            return_df = return_df
            return return_df
        elif len(return_df)<=30000:
            return_df = return_df.sample(n=round(len(return_df)/2), replace=False, random_state=rand_seed, axis=0)
            return return_df
        elif len(return_df)<=50000:
            return_df = return_df.sample(n=round(len(return_df)/3), replace=False, random_state=rand_seed, axis=0)
            return return_df
        elif len(return_df)>50000:
            return_df = return_df.sample(n=45000, replace=False, random_state=rand_seed, axis=0)
            return return_df
    else:
        print("Error")
        return None


def get_fcs_rand(pos_df,ncell_pos,rand_seed,guid_counts):
    chose_fcs_ncell= ncell_pos
    list_fcs=[]
    for fcs in guid_counts:
        if chose_fcs_ncell<= fcs[1] <= chose_fcs_ncell*3:
            list_fcs=np.append(list_fcs,fcs[0])
    if len(list_fcs)==0:
        for fcs in guid_counts:
            if chose_fcs_ncell<= fcs[1] <= chose_fcs_ncell*3.5:
                list_fcs=np.append(list_fcs,fcs[0])
    if len(list_fcs)==0:
        for fcs in guid_counts:
            if chose_fcs_ncell<= fcs[1] <= chose_fcs_ncell*4:
                list_fcs=np.append(list_fcs,fcs[0])
    if len(list_fcs)==0:
        for fcs in guid_counts:
            if chose_fcs_ncell<= fcs[1] <= chose_fcs_ncell*4.5:
                list_fcs=np.append(list_fcs,fcs[0])
    if len(list_fcs)==0:
        for fcs in guid_counts:
            if chose_fcs_ncell<= fcs[1] <= chose_fcs_ncell*5:
                list_fcs=np.append(list_fcs,fcs[0])
    ######################
    random.seed(rand_seed)
    rand_fcs = random.choice(list_fcs)
    return(rand_fcs)

####################
#######################################################################################
def get_sorted_input(num, ncell, neg_df, pos_df, filename, rand_seed,guid_counts,pos_guid_counts):
    ###########################################
    ##################################
    ###############################
    count=1
    labels=[]
    fraction=[]
    cell_accuracy=[]
    checkpoint=100
    ##########################################
    num_neg = round(num / 2)
    num_non_neg = num - num_neg
    num_pos_high = round(num_non_neg / 4)
    num_pos_low = round(num_non_neg / 4)
    num_pos_mrd = round(num_non_neg / 4)
    num_pos_lmrd = num - (num_neg + num_pos_high + num_pos_low + num_pos_mrd)
    #####################
    for i in range(num_neg):
        ###################
        fraction.append(0)
        tmp_df = get_neg_df(neg_df,rand_seed+i,guid_counts,0) 
        tmp_mega = tmp_df
        cell_accuracy.append(tmp_mega[tmp_mega['pred']==tmp_mega['Dx']].shape[0]/tmp_mega.shape[0])
        print(tmp_mega[tmp_mega['pred']==tmp_mega['Dx']].shape[0]/tmp_mega.shape[0])
        tmp_mega = tmp_mega.sort_values(['pred_0', 'pred_1'], ascending=[True, False])
        tmp_mega.to_pickle(filename+'df_'+str(count))
        labels.append(0)
        count += 1
        tmp_df = None
        if i % checkpoint == 0:
            print('Done '+str(count-1)+' out of '+str(num))
    #####################################################
    ############################
    for i in range(num_pos_high):
        np.random.seed(rand_seed+i)
        frac = np.random.choice(np.round(np.linspace(0.1, 1.0, 10000, endpoint=False), 6))
        fraction.append(frac)
        ncell_pos = int(np.round(frac*ncell))
        chosen_fcs = get_fcs_rand(pos_df,ncell_pos,rand_seed+i,pos_guid_counts)
        chosen_pos_df = pos_df[pos_df['case'] == chosen_fcs]
        tmp_neg_df = get_neg_df(neg_df,rand_seed+10000+i,guid_counts,ncell_pos)
        tmp_pos_df = chosen_pos_df.sample(n=ncell_pos, replace=False, random_state=rand_seed+i, axis=0)
        tmp_df = pd.concat([tmp_neg_df, tmp_pos_df])
        tmp_mega = tmp_df       
        cell_accuracy.append(tmp_mega[tmp_mega['pred']==tmp_mega['Dx']].shape[0]/tmp_mega.shape[0])
        print(tmp_mega[tmp_mega['pred']==tmp_mega['Dx']].shape[0]/tmp_mega.shape[0])
        tmp_mega = tmp_mega.sort_values(['pred_0', 'pred_1'], ascending=[True, False])
        tmp_mega.to_pickle(filename+'df_'+str(count))
        labels.append(1)
        count += 1
        tmp_df = None
        if i % checkpoint == 0:
            print('Done '+str(count-1)+' out of '+str(num))
    #####################################################
    ############################
    for i in range(num_pos_low):
        np.random.seed(rand_seed+i)
        frac = np.random.choice(np.round(np.linspace(0.05, 0.1, 10000, endpoint=False), 6))
        fraction.append(frac)
        ncell_pos = int(np.round(frac*ncell))
        chosen_fcs = get_fcs_rand(pos_df,ncell_pos,rand_seed+i,pos_guid_counts)
        chosen_pos_df = pos_df[pos_df['case'] == chosen_fcs]
        tmp_neg_df = get_neg_df(neg_df,rand_seed+20000+i,guid_counts,ncell_pos)
        tmp_pos_df = chosen_pos_df.sample(n=ncell_pos, replace=False, random_state=rand_seed+i, axis=0)
        tmp_df = pd.concat([tmp_neg_df, tmp_pos_df])
        tmp_mega = tmp_df       
        cell_accuracy.append(tmp_mega[tmp_mega['pred']==tmp_mega['Dx']].shape[0]/tmp_mega.shape[0])
        print(tmp_mega[tmp_mega['pred']==tmp_mega['Dx']].shape[0]/tmp_mega.shape[0])
        tmp_mega = tmp_mega.sort_values(['pred_0', 'pred_1'], ascending=[True, False])
        tmp_mega.to_pickle(filename+'df_'+str(count))
        labels.append(1)
        count += 1
        tmp_df = None
        if i % checkpoint == 0:
            print('Done '+str(count-1)+' out of '+str(num))
    #####################################################
    ############################
    for i in range(num_pos_mrd):
        np.random.seed(rand_seed+i)
        frac = np.random.choice(np.round(np.linspace(0.001, 0.05, 10000, endpoint=False), 6))
        fraction.append(frac)
        ncell_pos = int(np.round(frac*ncell))
        chosen_fcs = get_fcs_rand(pos_df,ncell_pos,rand_seed+i,pos_guid_counts)
        chosen_pos_df = pos_df[pos_df['case'] == chosen_fcs]
        tmp_neg_df = get_neg_df(neg_df,rand_seed+30000+i,guid_counts,ncell_pos)
        tmp_pos_df = chosen_pos_df.sample(n=ncell_pos, replace=False, random_state=rand_seed+i, axis=0)
        tmp_df = pd.concat([tmp_neg_df, tmp_pos_df])
        tmp_mega = tmp_df  
        cell_accuracy.append(tmp_mega[tmp_mega['pred']==tmp_mega['Dx']].shape[0]/tmp_mega.shape[0])
        print(tmp_mega[tmp_mega['pred']==tmp_mega['Dx']].shape[0]/tmp_mega.shape[0])
        tmp_mega = tmp_mega.sort_values(['pred_0', 'pred_1'], ascending=[True, False])
        tmp_mega.to_pickle(filename+'df_'+str(count))
        labels.append(1)
        count += 1
        tmp_df = None
        if i % checkpoint == 0:
            print('Done '+str(count-1)+' out of '+str(num))
    #####################################################
    ############################
    for i in range(num_pos_lmrd):
        np.random.seed(rand_seed+i)
        frac = np.random.choice(np.round(np.linspace(0.00005, 0.001, 10000, endpoint=True), 6))
        fraction.append(frac)
        ncell_pos = int(np.round(frac*ncell))
        chosen_fcs = get_fcs_rand(pos_df,ncell_pos,rand_seed+i,pos_guid_counts)
        chosen_pos_df = pos_df[pos_df['case'] == chosen_fcs]
        tmp_neg_df = get_neg_df(neg_df,rand_seed+40000+i,guid_counts,ncell_pos)
        tmp_pos_df = chosen_pos_df.sample(n=ncell_pos, replace=False, random_state=rand_seed+i, axis=0)
        tmp_df = pd.concat([tmp_neg_df, tmp_pos_df])
        tmp_mega = tmp_df  
        cell_accuracy.append(tmp_mega[tmp_mega['pred']==tmp_mega['Dx']].shape[0]/tmp_mega.shape[0])
        print(tmp_mega[tmp_mega['pred']==tmp_mega['Dx']].shape[0]/tmp_mega.shape[0])
        tmp_mega = tmp_mega.sort_values(['pred_0', 'pred_1'], ascending=[True, False])
        tmp_mega.to_pickle(filename+'df_'+str(count))
        labels.append(1)
        count += 1
        tmp_df = None
        if i % checkpoint == 0:
            print('Done '+str(count-1)+' out of '+str(num))

    ###################################################################################
    print("Done! Check HDF5 file for input. !!!!! Note: Shuffle at the Dataloader!!!!")
    ########################    
    return(labels, fraction, cell_accuracy)
############################
############################################################################################################################################################
############################################################################################################################################################

#Train and validation synthetic samples
neg_cells = pd.read_feather('../tmp/train_neg_cells_auto_with_pred')
pos_cells = pd.read_feather('../tmp/train_pos_cells_auto_with_pred')

######################################################################
#############################################

df_count_groupby = neg_cells.groupby('case')
guid_counts=[]
for fcs in neg_cells['case'].unique():
    if len(df_count_groupby.get_group(fcs))>1000:
        guid_counts.append([fcs,len(df_count_groupby.get_group(fcs))])
    
pos_df_count_groupby = pos_cells.groupby('case')
pos_guid_counts=[]
for fcs in pos_cells['case'].unique():
    pos_guid_counts.append([fcs,len(pos_df_count_groupby.get_group(fcs))])


labels_train,fraction_train,cell_accuracy = get_sorted_input(num_train, ncell, neg_cells, pos_cells, training_out_file, rand_seed,guid_counts,pos_guid_counts)
labels_valid,fraction_valid,cell_accuracy = get_sorted_input(num_valid, ncell, neg_cells, pos_cells, valid_out_file, rand_seed+91645,guid_counts,pos_guid_counts)
############################

pipeline = make_pipeline(labels_train, fraction_train, cell_accuracy)
pipeline = make_pipeline(labels_valid, fraction_valid, cell_accuracy)

joblib.dump(pipeline, '../tmp/synthetic_samples/Training_input_auto.sav')
joblib.dump(pipeline, '../tmp/synthetic_samples/Valid_input_auto.sav')
########################################################################
########################################################################

# Test synthetic samples

neg_cells = pd.read_feather('../tmp/test_neg_cells_auto_with_pred')
pos_cells = pd.read_feather('../tmp/test_neg_cells_auto_with_pred')

######################################################################
#############################################

df_count_groupby = neg_cells.groupby('case')
guid_counts=[]
for fcs in neg_cells['case'].unique():
    if len(df_count_groupby.get_group(fcs))>1000:
        guid_counts.append([fcs,len(df_count_groupby.get_group(fcs))])
    
pos_df_count_groupby = pos_cells.groupby('case')
pos_guid_counts=[]
for fcs in pos_cells['case'].unique():
    pos_guid_counts.append([fcs,len(pos_df_count_groupby.get_group(fcs))])

########################################################################
#############################################
labels_test,fraction_test,cell_accuracy = get_sorted_input(num_test, ncell, neg_cells, pos_cells, testing_out_file, rand_seed+91645,guid_counts,pos_guid_counts)
##################################################################################
pipeline = make_pipeline(labels_test, fraction_test, cell_accuracy)
##################################################################################
joblib.dump(pipeline, '../tmp/Testing_input_auto.sav')