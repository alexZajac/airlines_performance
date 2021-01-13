/* if you have an installation of npm run: npm i node-fetch
   else you can try it in a hosted JS environement running node
*/
const fetch = require("node-fetch");
const fs = require("fs");

const dataset_url = "https://www.transtats.bts.gov/DownLoad_Table.asp?Table_ID=259&Has_Group=3&Is_Zipped=0"
const headers = {
    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
    "accept-language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
    "cache-control": "max-age=0",
    "content-type": "application/x-www-form-urlencoded",
    "sec-ch-ua": "\"Google Chrome\";v=\"87\", \" Not;A Brand\";v=\"99\", \"Chromium\";v=\"87\"",
    "sec-ch-ua-mobile": "?0",
    "sec-fetch-dest": "document",
    "sec-fetch-mode": "navigate",
    "sec-fetch-site": "same-origin",
    "sec-fetch-user": "?1",
    "upgrade-insecure-requests": "1",
    "cookie": "__utmz=261918792.1608625085.3.2.utmcsr=google|utmccn=(organic)|utmcmd=organic|utmctr=(not%20provided); _ga=GA1.2.1137685886.1608626742; __utma=42773238.1137685886.1608626742.1608812165.1609026643.3; __utmz=42773238.1609026643.3.3.utmcsr=bts.dot.gov|utmccn=(referral)|utmcmd=referral|utmcct=/; ASPSESSIONIDCARSRDDB=LMDILGABLIMOKGINEMMEJNHN; __utmc=261918792; __utma=261918792.1584229812.1608563938.1609185361.1609235758.15; f5avraaaaaaaaaaaaaaaa_session_=FBAGJKHPHIDKPICCLNHMIOGODKPNBHFPMPGNJBFAJPIDGHOAFKILCJKJLIODBGCONFIDDDGKOFPFEBGCNNNADHPGMAKGCJMOFDKOEPOOPGBFNPAOELHGEBPPKODMEFAB"
};
const fetch_options = {
    "headers": headers,
    "referrer": "https://www.transtats.bts.gov/DL_SelectFields.asp",
    "referrerPolicy": "strict-origin-when-cross-origin",
    "body": "UserTableName=T_100_Domestic_Segment__U.S._Carriers&DBShortName=Air_Carriers&RawDataTable=T_T100D_SEGMENT_US_CARRIER_ONLY&sqlstr=+SELECT+DEPARTURES_SCHEDULED%2CDEPARTURES_PERFORMED%2CPAYLOAD%2CSEATS%2CPASSENGERS%2CFREIGHT%2CMAIL%2CDISTANCE%2CUNIQUE_CARRIER%2CAIRLINE_ID%2CUNIQUE_CARRIER_NAME%2CCARRIER_NAME%2CCARRIER_GROUP%2CORIGIN_AIRPORT_ID%2CORIGIN%2CORIGIN_CITY_NAME%2CORIGIN_STATE_ABR%2CDEST_AIRPORT_ID%2CDEST%2CDEST_CITY_NAME%2CDEST_STATE_ABR%2CAIRCRAFT_GROUP%2CAIRCRAFT_TYPE%2CAIRCRAFT_CONFIG%2CYEAR%2CMONTH%2CDISTANCE_GROUP%2CCLASS+FROM++T_T100D_SEGMENT_US_CARRIER_ONLY&varlist=DEPARTURES_SCHEDULED%2CDEPARTURES_PERFORMED%2CPAYLOAD%2CSEATS%2CPASSENGERS%2CFREIGHT%2CMAIL%2CDISTANCE%2CUNIQUE_CARRIER%2CAIRLINE_ID%2CUNIQUE_CARRIER_NAME%2CCARRIER_NAME%2CCARRIER_GROUP%2CORIGIN_AIRPORT_ID%2CORIGIN%2CORIGIN_CITY_NAME%2CORIGIN_STATE_ABR%2CDEST_AIRPORT_ID%2CDEST%2CDEST_CITY_NAME%2CDEST_STATE_ABR%2CAIRCRAFT_GROUP%2CAIRCRAFT_TYPE%2CAIRCRAFT_CONFIG%2CYEAR%2CMONTH%2CDISTANCE_GROUP%2CCLASS&grouplist=&suml=&sumRegion=&filter1=title%3D&filter2=title%3D&geo=All%A0&time=All%A0Months&timename=Month&GEOGRAPHY=All&XYEAR=2017&FREQUENCY=All&VarName=DEPARTURES_SCHEDULED&VarDesc=DepScheduled&VarType=Num&VarName=DEPARTURES_PERFORMED&VarDesc=DepPerformed&VarType=Num&VarName=PAYLOAD&VarDesc=Payload&VarType=Num&VarName=SEATS&VarDesc=Seats&VarType=Num&VarName=PASSENGERS&VarDesc=Passengers&VarType=Num&VarName=FREIGHT&VarDesc=Freight&VarType=Num&VarName=MAIL&VarDesc=Mail&VarType=Num&VarName=DISTANCE&VarDesc=Distance&VarType=Num&VarDesc=RampTime&VarType=Num&VarDesc=AirTime&VarType=Num&VarName=UNIQUE_CARRIER&VarDesc=UniqueCarrier&VarType=Char&VarName=AIRLINE_ID&VarDesc=AirlineID&VarType=Num&VarName=UNIQUE_CARRIER_NAME&VarDesc=UniqueCarrierName&VarType=Char&VarDesc=UniqCarrierEntity&VarType=Char&VarDesc=CarrierRegion&VarType=Char&VarDesc=Carrier&VarType=Char&VarName=CARRIER_NAME&VarDesc=CarrierName&VarType=Char&VarName=CARRIER_GROUP&VarDesc=CarrierGroup&VarType=Num&VarDesc=CarrierGroupNew&VarType=Num&VarName=ORIGIN_AIRPORT_ID&VarDesc=OriginAirportID&VarType=Num&VarDesc=OriginAirportSeqID&VarType=Num&VarDesc=OriginCityMarketID&VarType=Num&VarName=ORIGIN&VarDesc=Origin&VarType=Char&VarName=ORIGIN_CITY_NAME&VarDesc=OriginCityName&VarType=Char&VarName=ORIGIN_STATE_ABR&VarDesc=OriginState&VarType=Char&VarDesc=OriginStateFips&VarType=Char&VarDesc=OriginStateName&VarType=Char&VarDesc=OriginWac&VarType=Num&VarName=DEST_AIRPORT_ID&VarDesc=DestAirportID&VarType=Num&VarDesc=DestAirportSeqID&VarType=Num&VarDesc=DestCityMarketID&VarType=Num&VarName=DEST&VarDesc=Dest&VarType=Char&VarName=DEST_CITY_NAME&VarDesc=DestCityName&VarType=Char&VarName=DEST_STATE_ABR&VarDesc=DestState&VarType=Char&VarDesc=DestStateFips&VarType=Char&VarDesc=DestStateName&VarType=Char&VarDesc=DestWac&VarType=Num&VarName=AIRCRAFT_GROUP&VarDesc=AircraftGroup&VarType=Num&VarName=AIRCRAFT_TYPE&VarDesc=AircraftType&VarType=Char&VarName=AIRCRAFT_CONFIG&VarDesc=AircraftConfig&VarType=Num&VarName=YEAR&VarDesc=Year&VarType=Num&VarDesc=Quarter&VarType=Num&VarName=MONTH&VarDesc=Month&VarType=Num&VarName=DISTANCE_GROUP&VarDesc=DistanceGroup&VarType=Num&VarName=CLASS&VarDesc=Class&VarType=Char",
    "method": "POST",
    "mode": "cors"   
};
const output_filename = "dataset.zip"

(async () => {
    const res = await fetch(dataset_url, fetch_options);
    const fileStream = fs.createWriteStream(output_filename);
    await new Promise((resolve, reject) => {
        res.body.pipe(fileStream);
        res.body.on("error", reject);
        fileStream.on("finish", resolve);
      });
})();