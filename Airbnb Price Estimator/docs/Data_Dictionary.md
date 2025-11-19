# AB_NYC_2019 Data Dictionary

| Column Name                        | Data Type   | Description                                      | Example Values / Notes                                                                 |
|-----------------------------------|-------------|--------------------------------------------------|------------------------------------------------------------------------------------------|
| id                                | Integer     | Unique identifier for each listing               | 2539, 2595, 3647                                                                         |
| name                              | String      | Name of the listing                              | “Clean & quiet apt home by the park”                                                     |
| host_id                           | Integer     | Unique identifier for the host                   | 2787, 2845                                                                               |
| host_name                         | String      | Name of the host                                 | “John”, “Alice”                                                                          |
| neighbourhood_group               | Categorical | Broad region in NYC where the listing is located | “Brooklyn”, “Manhattan”, “Queens”, “Bronx”, “Staten Island”                              |
| neighbourhood                     | Categorical | Specific neighborhood within the region          | “Williamsburg”, “Midtown”, “Astoria”                                                     |
| latitude                          | Float       | Latitude coordinate of the listing               | 40.64749                                                                                 |
| longitude                         | Float       | Longitude coordinate of the listing              | -73.97237                                                                                |
| room_type                         | Categorical | Type of rental offered                           | “Entire home/apt”, “Private room”, “Shared room”, “Hotel room”                           |
| price                             | Integer     | Price per night in USD                           | Range: 0–10000, median ~149; may contain outliers                                        |
| minimum_nights                    | Integer     | Minimum number of nights for booking             | Range: 1–1000+, median ~3; extreme values may exist                                      |
| number_of_reviews                 | Integer     | Total number of reviews for the listing          | Range: 0–629, median ~6                                                                   |
| last_review                       | Date        | Date of the most recent review                   | “2019-06-23”, may have missing values                                                     |
| reviews_per_month                 | Float       | Average number of reviews per month              | Range: 0–20, may have missing values                                                      |
| calculated_host_listings_count    | Integer     | Number of listings the host has                  | Range: 1–327                                                                              |
| availability_365                  | Integer     | Number of days the listing is available per year | Range: 0–365, median ~112                                                                 |
