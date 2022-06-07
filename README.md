# Revision_of_MAML
Repositori amb el codi utilitzat per fer el treball "Més enllà de l’Aprenentatge Supervisat: Revisió de Models Agnòstics de Meta-Learning"

## MAML
El codi per a l'execució de les sinusoides de MAML que es troba dins *MAML/maml* ha estat extret de https://github.com/cbfinn/maml, afegint alguna modificació. El codi original utilitzava la versió de *TensorFLow 1.0*, la qual ja no era compatible amb les últimes versions de *Python*. Per tal de solucionar això, les últimes versions de *TensorFlow* contenten un mòdul anomenat *compat.v1*, el qual implementa algunes de les funcions bàsiques d'aquesta versió antiga de *TensorFlow*. Per facilitat d'ús, hem utilitzat aquest mòdul de la llibreria. A més per tal de poder guardar els resultats obtinguts amb la sinusoide i poder fer les gràfiques per poder visualitzar els resultats he creat una nova funció de test anomenada *my_test()* de tal manera que guarda un fitxer csv amb la informació necessària per fer els gràfics. Les instruccions d'execució del codi es troben al principi de tot del fitxer main.py. Tot i així, la comanda per executar-lo és:

**python main.py --datasource=sinusoid --logdir=logs/sine/ --metatrain_iterations=70000 --norm=None --update_batch_size=10**

on el paràmetre *update_batch_size* és el nombre de mostres d'entrenament.

## HowToTrainYourMAML
Aquest codi que es troba al fitxer *HowToTrainYourMAMLPytorch-master* ha estat extret de https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch. Aquest codi és el que ha sigut utilitzat per a la reproducció dels experiments de classificació **N-Way K-Shot**, variant els paràmetres N i K. Per tal d'executar el codi s'ha de crear un fitxer json i col·locar-lo al directori *experiment_config*, seguint la mateixa estructura que els fitxers que ja es troben allà. Aquest fitxer serà el que contindrà tots els paràmetres necessaris per a l'execució del model. Després, al directori *experiment_scripts* es troben els fitxers executables bash que criden als fitxers json i els passen com a paràmetres del script de *Python*. Utilitzant aquests scripts de bash ja creats es poden reproduir tots els experiments realitzats al treball per un problema de classificació utilitzant MAML.

## Few-Shot (Matching Networks i Prototypical Networks)
Aquest codi que es troba al directori *few-shot-master* ha estat extret de https://github.com/oscarknagg/few-shot. Implementa els algorismes de MAML, Matching Networks i Prototypical Networks. Per l'elaboració d'aquest treball, aquest codi ha estat utilitzat únicament per l'execució de les Matching i les Prototypical Networks. Per executar aquest codi es pot consultar el script de bash anomenat *exp_pipeline.sh* que he creat dins la carpeta *few-shot-master/experiments*, en el qual es troben el seguit d'experiments que he fet per tal d'avaluar l'eficàcia d'aquests models.

## Relation Networks
Aquest codi es troba al directori *LearningToCompare_FSL-master* i ha estat extret de https://github.com/floodsung/LearningToCompare_FSL. EL codi original està escrit en una versió anterior de *PyTorch*, aleshores per tal d'assegurar el seu correcte funcionament he hagut de modificar alguna de les línies de codi, per tal que funcioni bé en l'última versió. A més he afegit a l'algorisme un criteri de parada, en el qual si no obté cap millora en l'accuracy durant les últimes 10 fases de test donarem per acabat l'entrenament. Per executar aquest codi es pot consultar el script de bash anomenat *exp_pipeline.sh* que he creat dins la carpeta *LearningToCompare_FSL-master/miniimagenet*, en el qual es troben el seguit d'experiments que he fet per tal d'avaluar l'eficàcia d'aquest model.


Per tal de poder descarregar les bases de dades en l'estructura concreta que requereix cada algorisme descarregar la següent carpeta: https://uab-my.sharepoint.com/:f:/g/personal/1533660_uab_cat/EqaOVswEL3NFkJYbnmgF-5wBCI1vKt3mSsRLhVbD4HRBCw?e=7X7Geh. S'han de moure els arxius:
- *datas.zip* de la carpeta *LearningtoCompareRN* a la carpeta *LearningToCompare_FSL-master* i descomprimir el fitxer.
- *datasets* de la carpeta *HowToTrainMAML* a la carpeta *HowToTrainYourMAMLPytorch-master* i descomprimir tots els arxius **.zip** que hi ha dins.
- *data* de la carpeta *few-shot* a la carpeta *few-shot-master* i descomprimir tots els arxius **.zip** que hi ha dins.
