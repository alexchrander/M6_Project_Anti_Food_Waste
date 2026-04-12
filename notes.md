#### build_dataset.py
* The `offer_start_time`, `offer_end_time`, and `offer_last_updated` data fetched from the API is in UTC. The `fetched_at` variable in the same dataset is in CEST which means there's a 2 hour gap between the two. `fetched_at` is 2 hours ahead of the rest.
    * The same goes for the logged time stamps in the run_log.csv, it's in CEST and also 2 hours ahead of the rest.

>I recommend that we push `offer_start_time`, `offer_end_time`, and `offer_start_last_updated` data 2 hours so it fits CEST timezone. We should do that in the build_dataset.py or feature engineering.py.

* I've splitted aggregate_lifecycles() into two seperate functions: one that contains the features that can be used for the current table, and one for features that can't be used for the current table data, only the history table data.

#### General
* How does `offer_stock` work for products from the meat department / butcher's counter? Is each yellow label printed specifically for each product so they all have a different yellow label because of their different weights? 

#### evaluate.py
* Use these metrics for evaluation:
    * PR-AUC
    * F1, Precision, Recall
    * Log Loss

#### run_pipeline.py
* We should have a different threshold than 0.5 for the ml model to decide when it should say will_sell = true/false.
* Move the sell_threshold to the config.py in some way. The connection between fetching and ml pipeline needs rethinking. How will we structure main.py in the end?

#### App
* Use Streamlit in UCloud. Can connect directly to the vscode run so we can build everything directly inside the vm (vscode) UCloud run.
* Use caching for updating the app every 15 minutes. This stores the data for 15 minutes and then auto updates which is faster than auto reloading for every click which normally triggers a new query to the system.

#### Notes from group meeting
* Alexander pusher kode til github, så Peter kan rette sin kode til og får merged hele ML pipelinen.
* Derefter skal vi have besluttet os for hvilke kolonner, der skal med i predict tabellen i MySQL. Der beslutter vi efter at Peter har pushed sin kode.
    * Dertil skal vi have rettet predict.py scriptet, så det indsætter kolonner efter vores ønskede struktur i predict tabellen i MySQL.
* Alexander skriver til Primoz ang. main.py. Giver det mening at have med eller ej med vores struktur? 
* App.py laves i henhold til predict.py og predict tabellen fra MySQL. Alexander skal nok hjælpe med at give det indledende kode, så det henter rigtigt fra databasen.
* Begynd det indlende arbejde med vores projektskrivning. Start et overleaf projekt og få dokumenteret processer.
* Lav et shell script for predict.py magen til de andre + få sat automatisk cron kørsel op ligesom de andre to automatiske processer.
* Burde vi bruge wandb? For logging og visualisering af vores ml pipeline? 

#### Notes Anders
* Hvorfor har vi store lng og lat. med i numeric cols?
    * Vi tager store_id med som en katogorisk feature, og vi flytter store lng og lat. til kategorisk feature.
    * Kan man have nogle features til at flyde igennem ml- og prediction_pipeline uden at de nødvendigvis skal hjælpe med at træne ml modellen? Fordi så kan man bruge disse features som app features e.l., i stedet for at bruge dem til at træne modellen.
* Arbejde med at indarbejde sådan løbende opdateringsfeatures i lifecycle aggregeringen

#### Plan for nu
* Encoder.joblib? Laver vi encoder rigtigt? 
* Vi burde have et tidsbaseret train/test split i stedet for et random_state = 42.
* Er der data leakage ved at vores champion model bliver testet på et nyt test split? Altså, har den set mange af de tidligere offers i dens træningsfase dengang, som den så bliver testet på nu?
* Predict.py. Få scriptet til at virke i det directory den ligger i nu. Dertil, få besluttet hvilke endelige features vi sender til et prediction table i SQL. Er det alle features fetched fra API'en + alt det vi har feature engineered? 
* App.py. Få et app script til at virke. Indtil vi har fået et prediction table i SQL, kan man bruge /M6_Project_Anti_Food_Waste/test_data/predictions.csv


#### Alex noter
* Måske en sort by funktion i appen? Så man kan sortere på produkterne.
* Vise antal/vægt på hvert produkt? Hvor mange er der tilbage?
* Sell_through_rate burde måske bruge last_seen i stedet for final_stock, så det kan bruges til prediction også?
* Vi burde nok lave build_dataset logikken direkte i vores fetch e.l.

#### Vi mangler:
* Predictions shell file, cron job og cron log
* Peter retter småting i appen
* Skrive rapport
* Lave docker ting