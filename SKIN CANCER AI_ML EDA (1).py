#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
from PIL import Image


# In[2]:


skin_df = pd.read_csv(r"Desktop\dataverse_files\HAM.csv")


# In[3]:


skin_df.head()


# In[4]:


image_path = {os.path.splitext(os.path.basename(x))[0]: x
              for x in glob(os.path.join(r"Desktop\dataverse_files", '*', '*.jpg'))}


# In[5]:


skin_df['path'] = skin_df['image_id'].map(image_path.get)


# In[6]:


skin_df['image'] = skin_df['path'].map(lambda x: np.asarray(Image.open(x).resize((32,32))))


# In[7]:


print(skin_df['dx'].value_counts())


# In[8]:


lesion_type_dict = {
    'nv': 'Melanocytic_nevi',
    'mel': 'melanoma',
    'bkl': 'Benign_keratosis-like_lesions',
    'bcc': 'Basal_cell_carcinoma',
    'akiec': 'Actinic_keratoses',
    'vasc': 'Vascular_lesions',
    'df': 'Dermatofibroma'
}

lesion_danger = {
    'nv': 0, # 0 for benign
    'mel': 1, # 1 for malignant
    'bkl': 0, # 0 for benign
    'bcc': 1, # 1 for malignant
    'akiec': 1, # 1 for malignant
    'vasc': 0,
    'df': 0
}


# In[9]:


skin_df["cell_type"] = skin_df["dx"].map(lesion_type_dict.get)


# In[10]:


skin_df.head()


# In[11]:


skin_df.head()


# In[12]:


skin_df["Malignant"] = skin_df["dx"].map(lesion_danger.get)


# In[13]:


skin_df["Malignant"] = skin_df["dx"].map(lesion_danger.get)
skin_df.head()


# In[14]:


skin_df["cell_type_idx"] = pd.Categorical(skin_df["cell_type"]).codes
skin_df.sample(3)


# In[15]:


sns.set(style="whitegrid")

# Create a color palette
colors = sns.color_palette("pastel")

# Create a bar plot
fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))
skin_df["cell_type"].value_counts().plot(kind="bar", ax=ax1, color=colors, edgecolor="black")

# Add title and labels
ax1.set_title("Counts for Each Type of Lesions", fontsize=16)
ax1.set_xlabel("Cell Type", fontsize=14)
ax1.set_ylabel("Count", fontsize=14)

# Customize the x-axis labels for better readability
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right")

# Show the plot
plt.show()


# In[16]:


sns.set(style="whitegrid")

# Create a color palette
colors = sns.color_palette("coolwarm")

# Create a bar plot
plt.figure(figsize=(8, 6))
skin_df["Malignant"].value_counts().plot(kind="bar", color=colors, edgecolor="black")

# Add title and labels
plt.title("Benign vs Malignant", fontsize=16)
plt.xlabel("Malignancy", fontsize=14)
plt.ylabel("Count", fontsize=14)

# Customize x-axis labels
plt.xticks(ticks=[0, 1], labels=["Benign", "Malignant"], rotation=0)

# Show the plot
plt.show()


# In[17]:


skin_df["localization"].value_counts().plot(kind='bar', title="Location of Lesions")


# In[18]:


sns.set(style="whitegrid")

# Create a color palette
colors = sns.color_palette("husl", n_colors=len(skin_df["localization"].unique()))

# Create a bar plot
plt.figure(figsize=(12, 8))
skin_df["localization"].value_counts().plot(kind='bar', color=colors, edgecolor="black")

# Add title and labels
plt.title("Distribution of Lesion Localizations", fontsize=16)
plt.xlabel("Localization", fontsize=14)
plt.ylabel("Count", fontsize=14)

# Show the plot
plt.show()


# In[19]:


sns.set(style="whitegrid")

# Create a histogram
plt.figure(figsize=(12, 6))
skin_df["age"].hist(bins=50, color='skyblue', edgecolor='black')

# Add title and labels
plt.title("Distribution of Age", fontsize=16)
plt.xlabel("Age", fontsize=14)
plt.ylabel("Frequency", fontsize=14)

# Show the plot
plt.show()


# In[20]:


sns.set(style="whitegrid")

# Create a color palette
colors = sns.color_palette("viridis")

# Create a bar plot
plt.figure(figsize=(10, 6))
skin_df["dx_type"].value_counts().plot(kind='bar', color=colors, edgecolor='black')

# Add title and labels
plt.title("Distribution of Diagnosis Types", fontsize=16)
plt.xlabel("Diagnosis Type", fontsize=14)
plt.ylabel("Count", fontsize=14)

# Show the plot
plt.show()


# In[21]:


sns.set(style="whitegrid")

# Plot overall distribution of 'sex'
plt.figure(figsize=(12, 6))
skin_df["sex"].value_counts().plot(kind="bar", color=sns.color_palette("pastel"), edgecolor='black')
plt.title("Male vs Female (Overall)", fontsize=16)
plt.xlabel("Sex", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.show()


# In[22]:


plt.figure(figsize=(12, 6))
skin_df[skin_df["Malignant"] == 1]["sex"].value_counts().plot(kind="bar", color=sns.color_palette("pastel"), edgecolor='black')
plt.title("Male vs Female. Malignant Cases", fontsize=16)
plt.xlabel("Sex", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.show()


# In[23]:


from skimage.io import imread


# In[24]:


skin_df["image"] = skin_df["path"].map(imread) # read the image to array values


# In[25]:


skin_df.iloc[0]["image"] 


# In[26]:


skin_df["image"].map(lambda x: x.shape).value_counts() 


# In[27]:


n_samples = 5 
fig, m_axs = plt.subplots(7, n_samples, figsize=(4*n_samples, 3 * 7))

for n_axs, (type_name, type_rows) in zip(m_axs, skin_df.sort_values(["cell_type"]).groupby("cell_type")):
    n_axs[0].set_title(type_name)
    for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples, random_state=0).iterrows()):
        c_ax.imshow(c_row["image"])
        c_ax.axis("off")
fig.savefig("category_samples.png", dpi=300)


# In[28]:


rgb_info_df = skin_df.apply(lambda x: pd.Series({'{}_mean'.format(k): v for k, v 
                                                 in zip(["Red", "Blue", "Green"], 
                                                        np.mean(x["image"], (0, 1)))}), 1)


gray_col_vec = rgb_info_df.apply(lambda x: np.mean(x), 1) # take the mean value across columns of rgb_info_df
for c_col in rgb_info_df.columns:
    rgb_info_df[c_col] = rgb_info_df[c_col]/gray_col_vec 
rgb_info_df["Gray_mean"] = gray_col_vec
rgb_info_df.sample(3)


# In[29]:


for c_col in rgb_info_df.columns:
    skin_df[c_col] = rgb_info_df[c_col].values


# In[30]:


# let's draw a plot showing the distribution of different cell types over colors!
sns.pairplot(skin_df[["Red_mean", "Green_mean", "Blue_mean", "Gray_mean", "cell_type"]], 
             hue="cell_type", plot_kws = {"alpha": 0.5})


# In[31]:


n_samples = 5
for sample_col in ["Red_mean", "Green_mean", "Blue_mean", "Gray_mean"]:
    fig, m_axs = plt.subplots(7, n_samples, figsize=(4 * n_samples, 3 * 7))
    fig.suptitle(f"Change in cell type appearance as {sample_col} change")
    # define a function to get back a dataframe with 5 samples sorted by color channel values 
    def take_n_space(in_rows, val_col, n):
        s_rows = in_rows.sort_values([val_col])
        s_idx = np.linspace(0, s_rows.shape[0] - 1, n, dtype=int)
        return s_rows.iloc[s_idx]
    
    for n_axs, (type_name, type_rows) in zip(m_axs, skin_df.sort_values(["cell_type"]).groupby("cell_type")):
        for c_ax, (_, c_row) in zip(n_axs, take_n_space(type_rows, sample_col, n_samples).iterrows()):
            c_ax.imshow(c_row["image"])
            c_ax.axis("off")
            c_ax.set_title('{:2.2f}'.format(c_row[sample_col]))
        n_axs[0].set_title(type_name)
    fig.savefig("{}_samples.png".format(sample_col), dpi=300)


# In[32]:


from skimage.util import montage
rgb_stack = np.stack(skin_df.\
                     sort_values(['cell_type', 'Red_mean'])['image'].\
                     map(lambda x: x[::5, ::5]).values, 0)
rgb_montage = np.stack([montage(rgb_stack[:, :, :, i]) for i in range(rgb_stack.shape[3])], -1)
print(rgb_montage.shape)


# In[33]:


fig, ax1 = plt.subplots(1, 1, figsize = (20, 20), dpi=300)
ax1.imshow(rgb_montage)
fig.savefig('nice_montage.png')


# In[34]:


from skimage.io import imsave


# In[35]:


skin_df[['cell_type_idx', 'cell_type']].sort_values('cell_type_idx').drop_duplicates()


# In[36]:


from PIL import Image
def package_mnist_df(in_rows, 
                     image_col_name = 'image',
                     label_col_name = 'cell_type_idx',
                     image_shape=(28, 28), 
                     image_mode='RGB',
                     label_first=False
                    ):
    out_vec_list = in_rows[image_col_name].map(lambda x: 
                                               np.array(Image.\
                                                        fromarray(x).\
                                                        resize(image_shape, resample=Image.LANCZOS).\
                                                        convert(image_mode)).ravel())
    out_vec = np.stack(out_vec_list, 0)
    out_df = pd.DataFrame(out_vec)
    n_col_names =  ['pixel{:04d}'.format(i) for i in range(out_vec.shape[1])]
    out_df.columns = n_col_names
    out_df['label'] = in_rows[label_col_name].values.copy()
    if label_first:
        return out_df[['label']+n_col_names]
    else:
        return out_df


# In[37]:


from itertools import product

out_df = package_mnist_df(skin_df, 
                           image_shape=(32,32),
                           image_mode='RGB')
out_path = f'hmnist_{3}_{128}_RGB.csv'
out_df.to_csv(out_path, index=False)
print(f'Saved {out_df.shape} -> {out_path}: {os.stat(out_path).st_size/1024:2.1f}kb')


# In[ ]:





# In[ ]:




