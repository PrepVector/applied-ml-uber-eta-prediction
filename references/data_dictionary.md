# Data-dictionary

|Column|Description |
| :------------ |:---------------:|
|**ID**|order ID number| 
|**Delivery_person_ID**|Identifier of the delivery partner|
|**Delivery_person_Age**|Age of the delivery partner|
|**Delivery_person_Ratings**|Ratings of the delivery partner based on past deliveries|
|**Restaurant_latitude**| Latitude coordinate of the restaurant|
|**Restaurant_longitude**|Longitude coordinate of the restaurant|
|**Delivery_location_latitude**|Latitude coordinate of the delivery location|
|**Delivery_location_longitude**|Longitude coordinate of the delivery location|
|**Order_Date**|Date of the order|
|**Time_Ordered**|Time the order was placed|
|**Time_Order_picked**|Time the order was picked|
|**Weatherconditions**|Weather conditions of the day. The options are: Cloudy, Fog, NaN, Sandstorms, Stormy, Sunny, Windy|
|**Road_traffic_density**|Density of the traffic in the chosen city. The options are: High, Jam (indicating traffic jam), Low, Medium, NaN|
|**Vehicle_condition**|Delivery Partner's vehicle condition. The options in the front end maps to the following values in the dataset. They are: poor -->0, not bad--> 1, good--> 2, excellent--> 3|
|**Type_of_order**|The type of meal ordered by the customer. The options are: Buffet, Drinks, Meal, Snack|
|**Type_of_vehicle**|The type of vehicle delivery partner rides. The options are: Bicycle, Electric_scooter, Motorcycle, Scooter|
|**Multiple_deliveries**|Number of deliveries the driver has combined at the time or order. A maximum of 3 deliveries can be combined by the delivery partner. The options are: 0, 1, 2, 3, NaN|
|**Festival**|Ifs the current day a festival or not? The options are: Yes, No, NaN |
|**City**|Type of city. The options are: Metropolitan, NaN, Semi-Urban, Urban|
|**Time_taken(min)**| The time taken by the delivery partner to complete the order|
