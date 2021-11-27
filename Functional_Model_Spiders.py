#!/usr/bin/env python
# coding: utf-8

# <title>My New App Name</title>

# # ------------ Identificador de Arañas ------------

# In[1]:


import pathlib, platform
plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath


# !["Fuente: Wiki Commons"](Spider_Banner.jpg "Fuente: Wikimedia Commons")

# ## Bienvenidos a la herramienta de identificación de arañas! Sólo debes subir la imagen y darle al botón que dice "Identificar!"

# ##  <span style="color:#318CE7">Recomendaciones</span>

# * <span style="color:#1770c9">Una buena identificacion requiere multiples fotos! Si bien una foto puede ser más que suficiente, te recomendamos que uses al menos dos fotos para hacer de la identificación un proceso más fiable. Si el resultado no cambia, lo más probable es que la identificación automática sea correcta.</span>
# 
# * <span style="color:#318CE7">Una buena foto es clave para una correcta identificación! Al igual que la identificación hecha por un humano, una mala foto puede hacer la diferencia. Asegúrate de que la foto que uses tenga buena iluminación, y de ser posible recórtala para centrar a la araña que desees identificar (La foto de arriba es un buen ejemplo).</span>
# 
# * <span style="color:#1770c9">Solo usa fotos de arañas! Puede sonar muy obvio, pero si subes una foto de un gato te podría sorprender que el clasificador le asigna una identificación. Esto es porque se basa en probabilidades. (Aunque bien puedes hacer caso omiso, y divertirte subiendo imágenes de objetos que no sean arañas.) </span>
# 
# * <span style="color:#318CE7">Utiliza un buscador o iNaturalist para corroborar el resultado!. Cruzar los datos con fuentes externas (y validadas) es una buena práctica a la hora de identificar arañas.</span>
# 
# * <span style="color:#FFC300">El modo heurístico, es una modalidad experimental que fuerza la identificación, segmenta la imagen (al azar) para hacer detectiones por separado, y después promediar los resultados. Es una alternativa a recortar la imagen o usar otra foto. Si la identificación "tradicional" identificó a la araña, el modo heurístico tiende a dar el mismo resultado. Aparece un cuadrado abajo de la imagen principal en el segmento donde se identificó a la araña, si el cuadro está vacío o sólo captura ciertas características de la araña (patas, telaraña u otros objetos ajenos) lo más probable es que la identificación falló. Suele tardar entre 1 a 5 minutos.</span>

# In[2]:


from ipywidgets import widgets
from fastai.vision.all import PILImage
from fastai.vision.all import load_learner
from ipywidgets import VBox
import numpy as np
import io


# In[3]:


spider_class = load_learner('models/spider_class.pkl')


# In[4]:


upload = widgets.FileUpload(accept='image/*',
                            multiple=False)


# In[5]:


out_pl = widgets.Output()


# In[6]:


Image_W = widgets.Image(value=bytes(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x02\x08\x02\x00\x00\x00\x16\xe3!p\x00\x00\x00\x01sRGB\x00\xae\xce\x1c\xe9\x00\x00\x00\x04gAMA\x00\x00\xb1\x8f\x0b\xfca\x05\x00\x00\x00\tpHYs\x00\x00$\xe8\x00\x00$\xe8\x01\x82c\x05\x1c\x00\x00\x00\x10IDAT\x18Wc\x90\x91\x91a\x90\x91\x91\x01\x00\x02\xa8\x00\xa9\xc6\xbeAn\x00\x00\x00\x00IEND\xaeB`\x82'),
                        format='png',
                        width=1,
                        height=1)


# In[7]:


label = widgets.Label()
label.value = ''


# In[8]:


label_method = widgets.Label()
label_method.value = ''


# In[9]:


html_w = widgets.HTML()
html_w.value = ''


# In[10]:


btn_run = widgets.Button(description='Identificar!')
btn_run.style.button_color = 'orange'


# In[11]:


btn_heur = widgets.Button(description='Heurística!')
btn_heur.style.button_color = 'red'


# In[12]:


def random_soft_crop(img):
    w,h = img.size
    i = 0
    probs_list = []
    preds_list = []
    cords_list = []
    while i < 100:
        r_01 = np.random.uniform(0, w*.5)
        r_02 = np.random.uniform(0, h*.5)
        r_w = np.random.uniform(r_01*2, w)
        r_h = np.random.uniform(r_02*2, h)
        img_n = img.crop((r_01, r_02, r_w, r_h))
        img_n = PILImage(img_n)
        pred,indx,prob = spider_class.predict(img_n)
        percent_ = float(prob[indx]*100)

        if percent_ > 97.9:
            preds_list.append(pred)
            probs_list.append(percent_)
            cords_list.append((r_01, r_02, r_w, r_h))
        else:
            None
        i += 1
    
    return preds_list,probs_list, cords_list


# In[13]:


def random_hard_crop(img):
    w,h = img.size
    i = 0
    probs_list = []
    preds_list = []
    cords_list = []
    while i < 500:
        r_01 = np.random.uniform(0+w*.1, w*.7)
        r_02 = np.random.uniform(0+h*0.1, h*.7)
        r_w = np.random.uniform(r_01, w)
        r_h = np.random.uniform(r_02, h)
        img_n = img.crop((r_01, r_02, r_w, r_h))
        img_n = PILImage(img_n)
        pred,indx,prob = spider_class.predict(img_n)
        percent_ = float(prob[indx]*100)
        if percent_ > 97.7:
            preds_list.append(pred)
            probs_list.append(percent_)
            cords_list.append((r_01, r_02, r_w, r_h))
        else:
            None
        i += 1
    
    return preds_list,probs_list, cords_list


# In[14]:


def assign_category(preds, probs, cords):
    mask = np.array(preds) == max(set(preds))
    mean_prob = np.mean((np.array(probs)[mask]))
    mean_id = max(set(np.array(preds)[mask]))
    mean_cords = cords[np.argmax(preds)]
    
    return mean_id, mean_prob, mean_cords


# In[15]:


def on_click_classify(change):
    label.value = ''
    html_w.value= ''
    label_method.value = ''
    img = PILImage.create(upload.data[-1])
    out_pl.clear_output()
    Image_W.value = bytes(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x02\x08\x02\x00\x00\x00\x16\xe3!p\x00\x00\x00\x01sRGB\x00\xae\xce\x1c\xe9\x00\x00\x00\x04gAMA\x00\x00\xb1\x8f\x0b\xfca\x05\x00\x00\x00\tpHYs\x00\x00$\xe8\x00\x00$\xe8\x01\x82c\x05\x1c\x00\x00\x00\x10IDAT\x18Wc\x90\x91\x91a\x90\x91\x91\x01\x00\x02\xa8\x00\xa9\xc6\xbeAn\x00\x00\x00\x00IEND\xaeB`\x82')
    Image_W.width = 1
    Image_W.height = 1
    with out_pl: display(img.to_thumb(250,250))
    pred,indx,prob = spider_class.predict(img)
    percent_ = prob[indx]*100
    if percent_ >= 100:
        if pred in ['Latrodectus', 'Loxosceles']:
            label.value = f'Identificada como: {pred}; Probababilidad de Identificación Correcta: {percent_:.2f} %'
            html_w.value="<p style='color:#18A558'>Es de Importancia Médica!</p>"
        
        
        if pred in ['Leucage']:
            label.value = f'Identificada como: Leucauge; Probababilidad de Identificación Correcta: {percent_:.2f} %'
            html_w.value="<p style='color:#4169E1'>No es de Importancia Médica!</p>"
            
        else:
            label.value = f'Identificada como: {pred}; Probababilidad de Identificación Correcta: {percent_:.2f} %'
            html_w.value="<p style='color:#4169E1'>No es de Importancia Médica!</p>"
    else:
        label_method.value = 'Implementando Random Soft Cropping, espere'
        label.value = ''
        html_w.value= ''
        preds,probs,cords = random_soft_crop(img)
        if len(probs) > 1:
            category, category_probs, category_cords = assign_category(preds, probs, cords)
            if category_probs > 97.9:
                if category in ['Latrodectus', 'Loxosceles']:
                    crops = img.crop(category_cords).to_thumb(300,300)
                    
                    byteIO = io.BytesIO()
                    crops.save(byteIO, format='PNG')
                    byteArr = byteIO.getvalue()
                    Image_W.width = 100
                    Image_W.height = 150
                    Image_W.value = byteArr

                    label.value = f'Identificada como: {category}; Probababilidad de Identificación Correcta: {category_probs:.2f} %'
                    html_w.value="<p style='color:#18A558'>Es de Importancia Médica!</p>"
                    label_method.value = 'Encontrado con Random Soft Cropping'
                    
                
                elif category in ['Leucage']:
                    crops = img.crop(category_cords).to_thumb(300,300)
                    
                    byteIO = io.BytesIO()
                    crops.save(byteIO, format='PNG')
                    byteArr = byteIO.getvalue()
                    Image_W.width = 100
                    Image_W.height = 150
                    Image_W.value = byteArr
                    category = 'Leucauge'
                    
                    label.value = f'Identificada como: {category}; Probababilidad de Identificación Correcta: {category_probs:.2f} %'
                    html_w.value="<p style='color:#4169E1'>No es de Importancia Médica!</p>"
                    label_method.value = 'Encontrado con Random Soft Cropping'
                    
                else:
                    crops = img.crop(category_cords).to_thumb(300,300)
                    
                    byteIO = io.BytesIO()
                    crops.save(byteIO, format='PNG')
                    byteArr = byteIO.getvalue()
                    Image_W.width = 100
                    Image_W.height = 150
                    Image_W.value = byteArr
                    
                    label.value = f'Identificada como: {category}; Probababilidad de Identificación Correcta: {category_probs:.2f} %'
                    html_w.value="<p style='color:#4169E1'>No es de Importancia Médica!</p>"
                    label_method.value = 'Encontrado con Random Soft Cropping'
        else:
            label_method.value = 'Implementando Random Hard Cropping, espere'
            label.value = ''
            html_w.value= ''
            preds,probs,cords = random_hard_crop(img)
            if len(probs) > 1:
                category, category_probs, category_cords = assign_category(preds, probs, cords)
                if category_probs > 97.7:
                    if category in ['Latrodectus', 'Loxosceles']:
                        crops = img.crop(category_cords).to_thumb(300,300)
                        byteIO = io.BytesIO()
                        crops.save(byteIO, format='PNG')
                        byteArr = byteIO.getvalue()
                        Image_W.width = 100
                        Image_W.height = 150
                        Image_W.value = byteArr
                        label.value = f'Identificada como: {category}; Probababilidad de Identificación Correcta: {category_probs:.2f} %'
                        html_w.value="<p style='color:#18A558'>Es de Importancia Médica!</p>"
                        label_method.value = 'Encontrado con Random Hard Cropping'
                    
                    elif category in ['Leucage']:
                        crops = img.crop(category_cords).to_thumb(300,300)
                        byteIO = io.BytesIO()
                        crops.save(byteIO, format='PNG')
                        byteArr = byteIO.getvalue()
                        Image_W.width = 100
                        Image_W.height = 150
                        Image_W.value = byteArr
                        
                        category = 'Leucauge'
                        
                        label.value = f'Identificada como: {category}; Probababilidad de Identificación Correcta: {category_probs:.2f} %'
                        html_w.value="<p style='color:#4169E1'>No es de Importancia Médica!</p>"
                        label_method.value = 'Encontrado con Random Hard Cropping'
                    
                    
                    else:
                        crops = img.crop(category_cords).to_thumb(300,300)
                        byteIO = io.BytesIO()
                        crops.save(byteIO, format='PNG')
                        byteArr = byteIO.getvalue()
                        Image_W.value = byteArr
                        Image_W.width = 100
                        Image_W.height = 150
                        label.value = f'Identificada como: {category}; Probababilidad de Identificación Correcta: {category_probs:.2f} %'
                        html_w.value="<p style='color:#4169E1'>No es de Importancia Médica!</p>"
                        label_method.value = 'Encontrado con Random Hard Cropping'
            else:
                label.value = 'No Identificada'
                html_w.value="<p style='color:#18A558'>Use otra foto!</p>"
                label_method.value = 'Intente recortando la imagen! :D.'
        
        
btn_heur.on_click(on_click_classify)


# In[16]:


def on_click_classify(change):
    label.value = ''
    html_w.value= ''
    label_method.value = ''
    img = PILImage.create(upload.data[-1])
    out_pl.clear_output()
    Image_W.value = bytes(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x02\x08\x02\x00\x00\x00\x16\xe3!p\x00\x00\x00\x01sRGB\x00\xae\xce\x1c\xe9\x00\x00\x00\x04gAMA\x00\x00\xb1\x8f\x0b\xfca\x05\x00\x00\x00\tpHYs\x00\x00$\xe8\x00\x00$\xe8\x01\x82c\x05\x1c\x00\x00\x00\x10IDAT\x18Wc\x90\x91\x91a\x90\x91\x91\x01\x00\x02\xa8\x00\xa9\xc6\xbeAn\x00\x00\x00\x00IEND\xaeB`\x82')
    Image_W.width = 1
    Image_W.height = 1
    with out_pl: display(img.to_thumb(250,250))
    pred,indx,prob = spider_class.predict(img)
    percent_ = prob[indx]*100
    if percent_ >= 98.7:
        if pred in ['Latrodectus', 'Loxosceles']:
            label.value = f'Identificada como: {pred}; Probababilidad de Identificación Correcta: {percent_:.2f} %'
            html_w.value="<p style='color:#18A558'>Es de Importancia Médica!</p>"
        
        elif pred in ['Leucage']:
            label.value = f'Identificada como: Leucauge; Probababilidad de Identificación Correcta: {percent_:.2f} %'
            html_w.value="<p style='color:#4169E1'>No es de Importancia Médica!</p>"
        
        else:
            label.value = f'Identificada como: {pred}; Probababilidad de Identificación Correcta: {percent_:.2f} %'
            html_w.value="<p style='color:#4169E1'>No es de Importancia Médica!</p>"
    else:
        label.value = 'No Identificada'
        html_w.value="<p style='color:#18A558'>Use otra foto!</p>"
        label_method.value = 'Intente recortando la imagen! o use el modo Heurístico! :D.'
        
        
btn_run.on_click(on_click_classify)


# In[17]:


VBox([widgets.Label('Selecciona la foto de tu araña'), 
      upload, btn_run,btn_heur, out_pl, label, html_w, label_method, Image_W])


# ### Cómo se hizo?

# El clasificador se construyó a partir de los reportes de observaciones más frecuentes en el sitio de inaturalist.org para el país de México. Y distingue indiferentemente entre Familias y Géneros de Arañas.
# 
# * Familia **Anyphaenidae** (O _Arañas fantasma_) con **655** reportes de observación a la fecha*
# * Género **Araneus** (O _Arañas de jardín y granero_) con **377** reportes de observación a la fecha*
# * Género **Argiope** (O _Arañas tejedoras de Jardín_) con **4909** reportes de observación a la fecha*
# * Género **Cheiracanthium** con **219** reportes de observación a la fecha*
# * Género **Curicaberis** (O _Arañas cazadoras del Dios del Fuego_) con **655** reportes de observación a la fecha*
# * Familia **Dysderidae** (O _Arañas cazadoras_) con **276** reportes de observación a la fecha*
# * Género **Heteropoda** con **631** reportes de observación a la fecha*
# * Género **Kukulcania** con **1068** reportes de observación a la fecha*
# * Género **Latrodectus** (O _Viudas_) con **3860** reportes de observación a la fecha*
# * Género **Leucauge** (O _Arañas de Rayas Blancas_) con **2910** reportes de observación a la fecha*
# * Género **Loxosceles** (O _Arañas violinistas_) con **694** reportes de observación a la fecha*
# * Familia **Lycosidae** (O _Arañas Lobo_) con **7305** reportes de observación a la fecha*
# * Género **Micrathena** (O _Arañas Espinosas_) con **1227** reportes de observación a la fecha*
# * Género **Neoscona** (O _Arañas Tejedoras Manchadas_) con **7378** reportes de observación a la fecha*
# * Género **Oxyopes** (O _Arañas Lince_) con **429** reportes de observación a la fecha*
# * Género **Peucetia** (O _Arañas Lince Verde_) con **3726** reportes de observación a la fecha*
# * Familia **Pholcidae** (O _Arañas patonas_) con **2826** reportes de observación a la fecha*
# * Familia **Salticidae** (O _Arañas Saltarinas_) con **17293** reportes de observación a la fecha*
# * Familia **Scytodidae** (O _Arañas Escupidoras_) con **1172** reportes de observación a la fecha*
# * Género **Selenops** (O _Arañas de Pared_) con **1158** reportes de observación a la fecha*
# * Género **Steatoda** (O _Arañas falsas viudas_) con **1910** reportes de observación a la fecha*
# * Género **Tetragnatha** (O _Arañas de quelíceros alargados_) con **703** reportes de observación a la fecha*
# * Familia **Theraphosidae** (O _Tarántulas_) con **6214** reportes de observación a la fecha*
# * Familia **Thomisidae** (O _Arañas Cagrejo_) con **4043** reportes de observación a la fecha*
# * Género **Trichonephila** con **2542** reportes de observación a la fecha*
# * Familia **Zoropsidae** (O _Falsas arañas lobo_) con **1114** reportes de observación a la fecha*

# La distinción entre familias y géneros no es arbitraria. Es de interés por ejemplo, distinguir entre una **Steatoda** (o _falsa viuda_) y una **Latrodectus** (O _Viuda_), a pesar de que ambas son de la familia **Theridiidae**. En cambio dinstinguir entre las 177 especies reportadas a la fecha\* en México de la Familia **Salticidae** (O _Arañas saltarinas_) carece de interés para el propósito de este clasificador, también, por ejemplo el distinguir entre las 80 especies reportadas hasta la fecha\* de la Familia **Theraphosidae** (_Tarántulas_). Construir dicho clasificador para tales fines de distinción meticulosa sigue otros propósitos diferentes a éste.
# 
# Entre las familias y géneros excluidos por este clasificador se encuentra el Género **Olios** (Familia _Sparassidae_ al igual que el Género **Heteropoda**), ya que en México solo tiene 169 reportes de observación a la fecha\*, lo mismo pasa con el Género **Ariamnes** con 17 reportes de observación a la fecha\*. También fue excluido el Género **Phoneutria** (_Arañas bananeras_), que a pesar de ser arañas de importancia médica y tener reportes de observación en Honduras y Belice (Países vecinos), en México tiene 0 reportes de observación a la fecha\*, igualmente con el género **Sicarius** (_Arañas areneras_), con reportes de observación en países Sudamericanos, pero 0 a la fecha\* en México.

# El clasificador se hizo en una red neuronal convolucional de 34 capas (ResNet34) pre-entrenada (Dataset: Imagenet) y re-entrenada en las ultimas capas, con la libreria para Python: **FastAi**. La página web fue desplegada usando Jupyter Notebook y Voilà, y está alojada en **Heroku**.

# ###### *21 de Noviembre del 2021

# ##  <span style="color:#913831">Advertencia!</span>

# * <span style="color:#913831">Este identificador no sustituye la opinión de un experto ni da un diagnóstico médico</span>
# * <span style="color:#913831">Este identificador es falible y no incluye todas las familias, géneros, o especies de arañas que se han observado en México</span>
# * <span style="color:#913831">Este identificador funciona como herramienta auxiliar en la identificación de arañas.</span>
