# Riconoscimento_esercizi_per_anziani



Per il codice originale consultare il seguente link gitHub: https://github.com/eifrank/cv-fitness.

Questo progetto è testato su Windows 10.



Il sistema è in grado di riconoscere sei tipi di azioni: ['shoulders', 'foldedLegs', 'legs', 'jumpingJack', 'arms','squat'].




I 5 script principali sono sotto src, sono denominati in base all'ordine di esecuzione:

src/s1_get_skeletons_from_training_imgs.py \   
src/s2_put_skeleton_txts_to_a_single_txt.py  \
src/s3_preprocess_features.py \
src/s4_train.py \
src/s5_run.py \

I primi quattro script sono presenti all'interno del repository sopra citato. 




<br></br>
**Come eseguire lo script** 

Lo script src / s5_run.py serve per il riconoscimento delle azioni in tempo reale.

*Prova su file video*: \
python src/s5_run.py \
    --model_path model/trained_classifier.pickle \
    --data_type video \
    --data_path data_test/exercise.avi \
    --output_folder output
    
*Prova su una cartella di immagini*: \
python src/s5_run.py \
    --model_path model/trained_classifier.pickle \
    --data_type folder \
    --data_path data_test/apple/ \
    --output_folder output
    
*Prova sulla webcam*: \
python src/s5_run.py \
    --model_path model/trained_classifier.pickle \
    --data_type webcam \
    --data_path 0 \
    --output_folder output
    

*Link al dataset personalizzato*:  https://drive.google.com/file/d/1g6jBw2GowwCSWbWct0YbORafmxVvfC01/view?usp=sharing

Decomprimere il file e sostituire la cartella source_images3 estratta con quella presente in data/source_images3.

All'interno della cartella si trova il file valid_images.txt, che descrive l'etichetta di ogni immagine che abbiamo usato per l'allenamento. (puoi visualizzarlo su data / source_images3 / valid_images.txt .)



<br></br>
**Come eserguire il training**

Esegui i seguenti script uno per uno: \
python src/s1_get_skeletons_from_training_imgs.py \
python src/s2_put_skeleton_txts_to_a_single_txt.py \
python src/s3_preprocess_features.py \
python src/s4_train.py \
python src/s5_run.py 


Al termine dell'addestramento, e' possibile eseguire lo script src/s5_run.py
