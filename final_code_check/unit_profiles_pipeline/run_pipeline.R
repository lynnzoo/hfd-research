#Written by Erin Kreus

#Set the working Directory
this_dir <- function(directory)
setwd(file.path(getwd(),directory))

#Load all packages
source("automated/load_dependencies_profiles.R")

ambulances <- function(vehicle_list) {
    #Written by: Erin Kreus
    #This function references a list of unique Ladder units that we build profiles for
    #and creates an Rmarkdown PDFs for profiles of each ladder unit.
    #Input:
    #ladder_list: file containing Ladder Units
    #Output:
    #Unit Profile PDFs for each Ladder Unit 
  
  ambulance_list <- vehicle_list %>% filter(unit_type == "Ambulance")
  for (i in unique(ambulance_list$unit)) {
      rmarkdown::render("AmbulanceProfileFormat.rmd", 
                        params = list(Unit_Interested = i),
                        output_file=paste0(i, "_UnitProfile.pdf"))
    }
  }
  
  
##Create Engine Function
engines <- function(engine_list) {
  #Written by: Erin Kreus
  #This function references a list of unique Engine units that we build profiles for
  #and creates an Rmarkdown PDFs for profiles of each engine unit.
  #Input:
  #vehicle_list: file containing list of all units
  #Output:
  #Unit Profile PDFs for each Engine Unit 
  
  engine_list <- vehicle_list %>% filter(unit_type == "Engine")
  for (i in unique(engine_list$unit)) {
    rmarkdown::render("EngineProfileFormat.rmd", 
                      params = list(Unit_Interested = i),
                      output_file=paste0(i, "_UnitProfile.pdf"))
  }
}

##Create Ladder Function
ladders <- function(vehicle_list) {
  #Written by: Erin Kreus
  #This function references a list of unique Ladder units that we build profiles for
  #and creates an Rmarkdown PDFs for profiles of each ladder unit.
  #Input:
  #vehicle_list: file containing list of all units
  #Output:
  #Unit Profile PDFs for each Ladder Unit 
  
  ladder_list <- vehicle_list %>% filter(unit_type == "Ladder")
  for (i in unique(ladder_list$unit)) {
    rmarkdown::render("LadderProfileFormat.rmd", 
                      params = list(Unit_Interested = i),
                      output_file=paste0(i, "_UnitProfile.pdf"))
  }
}

##Create Medic Function
medics <- function(vehicle_list) {
  #Written by: Erin Kreus
  #This function references a list of unique Medic units that we build profiles for
  #and creates an Rmarkdown PDFs for profiles of each Medic unit.
  #Input:
  #vehicle_list: file containing list of all units
  #Output:
  #Unit Profile PDFs for each Medic Unit 
  medic_list <- vehicle_list %>% filter(unit_type == "Medic")
  for (i in unique(medic_list$unit)) {
    rmarkdown::render("MedicProfileFormat.rmd", 
                      params = list(Unit_Interested = i),
                      output_file=paste0(i, "_UnitProfile.pdf"))
  }
}

#Run the pipeline
run_pipeline <- function(unitprofilesa, informationforcodes, vehicle_list,
                         unitprofilese,correctresponsese,
                         unitprofilesm,
                         unitprofilesl,correctresponsesl){
  vehicle_list_df <- read.csv(vehicle_list)
  ambulances(vehicle_list_df)
  ladders(vehicle_list_df)
  engines(vehicle_list_df)
  medics(vehicle_list_df)
}

run_pipeline(unitprofilesa = "ambulances_data_unitprofiles.csv", 
             informationforcodes = "subdividedcodes.csv", 
             vehicle_list="unit_list.csv",
             unitprofilese="engines_data_unitprofiles.csv",
             correctresponsese="engines_data_unitprofiles_correct.csv",
             unitprofilesm="medics_data_unitprofiles.csv",
             unitprofilesl="ladders_data_unitprofiles.csv",
             correctresponsesl="ladders_data_unitprofiles_correct.csv")
