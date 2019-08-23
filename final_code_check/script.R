#!/usr/bin/Rscript
#######################
## title: "Running Pipeline for HFD Choropleth Maps"
## author: "Shannon Chen"
#######################

## Automatically detect the file path of this file and set the working directory the to folder where this file is located
this_dir <- function(directory)
setwd( file.path(getwd(), directory) )
# setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
## Print out the current working directory 
getwd()

## Add the names of all the packages that you used in the pipeline to list.of.packages
list.of.packages <- c("cartography", "rgdal", "gsubfn", "plyr", "rstudioapi")

## Automatically install any necessary packages that have not already been installed
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)

## load all the packages that you used in the pipeline here
library("cartography")
library("rgdal")
library("gsubfn")
library("plyr")
library("rstudioapi")



#######################
## title: "Creating the Choropleth Maps for HFD"
## author: "Shannon Chen"
#######################


################### Median EMS Response Times ###################

yearextract <- function(df, yr) {
  df_year <- as.data.frame(df[which(df$year == yr),])
  return(df_year)
}

#mutates the dataframe to create intervals used in the choropleth map. 
breaksplit <- function(df, breaks, idx){
  df <- as.data.frame(df)
  df$fac <- as.numeric(df[,idx])
  cutintervals <- cut(df$fac, breaks)
  ranges <- cbind(as.numeric( sub("\\((.+),.*", "\\1", cutintervals) ),
                  as.numeric( sub("[^,]*,([^]]*)\\]", "\\1", cutintervals) ))
  df$fac <- ranges[,2]
  return(df)
}


################### Slow EMS ###################

slowems <- function(data, time){
  slow <- data[which(data$responsetime > time),]
  return(slow)
}

slowambulance <- function(data){
  data$unit <- as.character(data$unit)
  slowunits <- data[which(startsWith(data$unit, "A")==TRUE | startsWith(data$unit, "M")==TRUE),]
  return(slowunits)
}

slowcounting <- function(slowdf, col){
  slowcounted <- count(slowdf, col)
  return(slowcounted)
}

################### Incident Densities ###################

incidentyears <- function(data, yr){
  data$eventnum <- as.numeric(data$eventnum )
  incidentyr <- data[which(data$eventnum >= (yr * 100000000) & data$eventnum < ((yr + 1)*100000000)), ]
  return(incidentyr)
}


incidentcounts <- function(data, colname){ #column name in string 
  incidentcount <- count(data, colname)
  incidentcount[,1] <- as.numeric(unlist(incidentcount[,1]))
  incidentcount[,2] <- as.numeric(unlist(incidentcount[,2]))
  incident_grouped <- cbind(incidentcount[,1], incidentcount[,2])
  colnames(incident_grouped) <- c("Station", "Count")
  incident_grouped[1,1] <- 1 #get rid of station 0 because it's useless
  return(incident_grouped)
}


################### Busy Fractions ###################

busyfractionfixed <- function(data){ #Formats the busy fraction to fit the shapefile
  data[which(data$X==23),1] <- 22
  return(data)
}

################### Choropleth Plotting Function ###################

choroplot <- function(shape, data, variable, brk, color, legendtitle, maintitle, png_name){
  png(paste0("./figures/",png_name), width = 8, height = 7, units = 'in', res = 300)
  plot(shape, border = NA, col = NA, bg = "#A6CAE0", xlim=c(2891000, 3353840) , ylim= c(13750000, 14038770), main = maintitle)
  choroLayer(spdf = shape, df = data, var = variable, 
             breaks = brk, col = color, 
             border = "grey40", legend.pos = "right", lwd = 0.5, 
             legend.title.txt = legendtitle, 
             legend.values.rnd = 2, add = TRUE)
  dev.off()
}

#######################
## title: "Pipeline for HFD Choropleth Maps"
## author: "Shannon Chen"
#######################

run_pipeline <- function(shapefile_dsn, shapefile, med_ems_file, ems_file, incidents_file, distress_file, helping_file, chain_file){
  #Shapefile of HFD Station Jurisdictions
  HFD_jurs <- readOGR(dsn=shapefile_dsn, layer=shapefile)
  HFD_jurs <- HFD_jurs[-1, ] #Removed station 0 for aesthetic reasons
  HFD_jurs[which(HFD_jurs$D1==23),1] <- 22 #Changed to 22 because R doesn't recognize 22 as having a jurisdiction
  
  #Raw files that were uploaded
  med_ems <- read.csv(med_ems_file)
  ems <- read.csv(ems_file)
  incidents <- read.csv(incidents_file)
  distress <- read.csv(distress_file)
  helping <- read.csv(helping_file)
  averagechains <- read.csv(chain_file)
  
  ################### Median EMS Maps ###################
  
  medems12 <- yearextract(med_ems, 2012)
  medems16 <- yearextract(med_ems, 2016)
  medems17 <- yearextract(med_ems, 2017)
  write.csv(medems12, "./data/median_ems2012.csv")
  write.csv(medems16, "./data/median_ems2016.csv")
  write.csv(medems17, "./data/median_ems2017.csv")
  
  colsmed <- carto.pal(pal1 = "turquoise.pal", n1 = 5) #Median EMS Color Scheme
  
  medems12_fac <-breaksplit(medems12, c(0, 5, 6, 7, 10, 30), 3) 
  medems16_fac <-breaksplit(medems16, c(0, 5, 6, 7, 10, 30), 3)
  medems17_fac <-breaksplit(medems17, c(0, 5, 6, 7, 10, 30), 3)
  write.csv(medems12_fac, "./data/medems2012factorized.csv")
  write.csv(medems16_fac, "./data/medems2016factorized.csv")
  write.csv(medems17_fac, "./data/medems2017factorized.csv")
  
  
  choroplot(HFD_jurs, medems12_fac, "fac", c(5, 6, 7, 10, 30), colsmed, "Median Minutes", "2012 Response Times", "figure10a.png")
  choroplot(HFD_jurs, medems16_fac, "fac", c(5, 6, 7, 10, 30), colsmed, "Median Minutes", "2016 Response Times", "figure10b.png")
  choroplot(HFD_jurs, medems16_fac, "fac", c(5, 6, 7, 10, 30), colsmed, "Median Minutes", "2016 Response Times", "figure11b.png")
  choroplot(HFD_jurs, medems17_fac, "fac", c(5, 6, 7, 10, 30), colsmed, "Median Minutes", "2017 Response Times", "figure12.png")
  
  ################### Slow EMS Map ###################
  colslow <- carto.pal(pal1 = "sand.pal", n1 = 6) #Slow Ambulances Color Palette
  slowstuff <- slowems(ems, 10)
  write.csv(slowstuff, "./data/slowems.csv")
  
  slow_amb <- slowambulance(slowstuff)
  write.csv(slow_amb, "./data/slowunits.csv")
  
  slowcount <- slowcounting(slow_amb, "incident_juris")
  write.csv(slowcount, "./data/slowunitscount.csv")
  
  slow_grouped <- breaksplit(slowcount, c(0, 2500, 4500, 7500, 12000, 18000), 2)
  write.csv(slow_grouped, "./data/slowresponsefactorized.csv")
  
  choroplot(HFD_jurs, slow_grouped, "fac", c(2500, 4500, 7500, 12000, 18000), colslow, "EMS Dispatches\n2011-18", "Slow Response Counts (>10 Minutes)", "figure13.png")
  
  ################### Incident Density Maps ###################
  
  colsincidents <- carto.pal(pal1 = "red.pal", n1 = 6) #Incident Map Color Palette
  incidents12 <- incidentyears(incidents, 12)
  incidents16 <- incidentyears(incidents, 16)
  write.csv(incidents12, "./data/incidents2012.csv")
  write.csv(incidents16, "./data/incidents2016.csv")
  
  incidents12_grouped <- incidentcounts(incidents12, "incident_juris")
  incidents12split <- breaksplit(incidents12_grouped, c(0, 1000, 2000, 3000, 5000, 7000, 10000), 2)
  incidents16_grouped <- incidentcounts(incidents16, "incident_juris")
  incidents16split <- breaksplit(incidents16_grouped, c(0, 1000, 2000, 3000, 5000, 7000, 10000), 2)
  write.csv(incidents12split, "./data/incidents2012factorized.csv")
  write.csv(incidents16split, "./data/incidents2016factorized.csv")
  
  choroplot(HFD_jurs, incidents12split , "fac", c(1000, 2000, 3000, 5000, 7000, 10000), colsincidents , "Counts", "2012 All Incidents", "figure7a.png")
  choroplot(HFD_jurs, incidents16split , "fac", c(1000, 2000, 3000, 5000, 7000, 10000), colsincidents , "Counts", "2016 All Incidents", "figure7b.png")
  choroplot(HFD_jurs, incidents16split , "fac", c(1000, 2000, 3000, 5000, 7000, 10000), colsincidents , "Counts", "2016 All Incidents", "figure11a.png")
  
  
  ################### Busy Fractions Maps ###################
  
  colsbusy <- carto.pal(pal1 = "orange.pal", n1 = 5) #Busy Fraction Color Palette
  
  distress_grouped <- breaksplit(distress, c(0, 0.2, 0.3, 0.4, 0.6, 0.7), 2)
  write.csv(distress_grouped, "./data/distressfractiongrouped2018.csv")
  
  helpingfixed <- busyfractionfixed(helping)
  helping_grouped <- breaksplit(helpingfixed, c(0, 0.2, 0.3, 0.4, 0.6, 0.7), 2)
  write.csv(helping_grouped, "./data/helpingfractiongrouped2018.csv")
  
  choroplot(HFD_jurs,  distress_grouped, "fac", c(0.2, 0.3, 0.4, 0.6, 0.7), colsbusy, "2018", "Distress Fractions", "distress_fractions2018.png")
  choroplot(HFD_jurs,  helping_grouped, "fac", c(0.2, 0.3, 0.4, 0.6, 0.7), colsbusy, "2018", "Helping Fractions", "figure14.png")
  
  ################## Average Chain Length Maps ########################
  
  colschain <- carto.pal(pal1="turquoise.pal", n1=5) #Average Chain Length Color Palette
  
  avchain_grouped <- breaksplit(averagechains, c(0, 1.12, 1.20, 1.28, 1.36, 1.44), 2)
  write.csv(avchain_grouped, "./data/averagechainsfactorized.csv")
  
  choroplot(HFD_jurs, avchain_grouped, "fac", c(1.12, 1.20, 1.28, 1.36, 1.44), colschain, "Chain Analaysis", "Average Chain Length", "figure22.png")
}

## call the driver function to run the entire automated data wrangling pipeline
hfd_maps <- run_pipeline(shapefile_dsn = "data/raw", shapefile="Still_Alarms_012319", med_ems_file = "data/median_by_station.csv", 
                         ems_file= "data/data_ems.csv", incidents_file = "data/incident_jurisdictions.csv", distress_file = "data/distress_fractions.csv", 
                         helping_file = "data/helping_fractions.csv", chain_file="data/average_chain_lengths.csv")

