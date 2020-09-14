# Bill Passage Team 3

## ACLU Data Notes

American Civil Liberties Union's (ACLU) mission is protecting civil liberties of all citizens through advocacy efforts against legislation that infringe on civil liberties, and ensuring that necessary statutes are in place to protect civil liberties.  To that end, one of the most important aspects of their work is monitoring legislation passed into law at national level and state level. Currently, this process of bill monitoring entails signifcant manual effort. Therefore, ACLU's state affiliates either, spend their time reading through all bills to identify legilation that they should oppose (or support), or end up having to pick and choose bills they read using their domain knowledge (e.g. picking bills sponsored by certain legislators/committees). The former takes away their time that could be spent on devising interventions (e.g. drafting lawsuits) and the latter can result in missing important legislation pieces.  

The goal of this project is to help ACLU better target their resources by identifying bills that are likely to be passed into law. The data for this project comes from LegiScan, an organization that collects and diseminates data about national and state level legislation and the legislative processes. LegiScan provides an API to acquire continuous updates to the legislative sessions for all 52 states and districts/territories of the US. Therefore, this project uses data that is publicly available. The provided database contains legilative session and bill data for the last ~10 years. The initial raw data was provided by LegiScan as a collection of JSON files. A cleaned schema from the JSON files was created, and includes the following infomation:
- *State legislative session information*: Information about legislative sessions for each session is available in `.sessions` with a unique session identifier `session_id`. Note that legislative sessions can be either periodic regular sessions or special sessions convened by the Governor. The boolean column `special` indicates whether the session is a special session or not. 

- *Bill information*: Data about bills introduced in each session is provided in `ml_policy_class.bills`. Each bill has a unique identifier `bill_id`. Information about the type of the bill, the session it belongs to, bill subjects, the date and the chamber/ body (House/Senate) where the bill was introduced are provided. The `url` field contains the link to the LegiScan webpage that shows the bill, if needed. 

- *Bill text information* : Bills typically get amended and go through different versions when it moves through the legislative process. The table `ml_policy_class.bill_texts` contains text and information about different versions of a bill. Each version has a unique identifier named `doc_id`. Each bill version/text has a type to indicate which form the bill is in (e.g. is it as it was introduced? amendments adopted?) and is indicated by the `type_id`. The title of the bill and a summary of the bill is given in the fields `bill_title` and `bill_description`, respectively. The `url` fields point to the LegiScan page and the state's webpage for the bill text.

- *People information*: Information about congress members (both senate and house) related to each session is available in `ml_policy_class.session_people`. Each member has a unique identifier `person_id` and information such as their party affiliation, role is provided. Further, several identifiers for external data sources such as _follow the money_, _votesmart_, and _opensecrets_ are available as well. In addition, `ml_policy_class.bill_sponsors` table contains information about people who sponsored each bill.

- *Voting information*: Voting information for roll calls of bills is available in the table `ml_policy_class.bill_votes`. Each roll call has a unique identifier `vote_id` and contains a summary of the votes. Note that if individual person votes for each roll call can be obtained through the LegiScan API (API info provided below) using the roll call id. 

- *Bill event information*: Events that a bill goes through and the status progress of a bill are available in the tables `ml_policy_class.bill_events` and `ml_policy_class.bill_progress` respectively. 

- *Amendment information*: Information about amendments made to the bill from the house and the seante is available in the table `ml_policy_class.bill_amendments`. The links to find the text for the amendment (from the Legiscan website and the State website) are given in the columns `legiscan_url, state_url`. 

- *catalog tables*: In addition the the `ml_policy_class` schema, a `catalogs` schema that contains mappings from LegiScan identifiers to descriptions. For instance, identifiers such as party ids, bill text type ids, status codes are mapped to their descriptions in this schema.

- *Legiscan API for additional data*: LegiScan API (https://legiscan.com/legiscan) can be used to acquire more data about bills if necessary and the user manual for the API can be found here: https://legiscan.com/gaits/documentation/legiscan. 


