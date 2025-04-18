from typing import list
import pandas as pd
from brendapyrser import BRENDA
from brendapyrser import Reaction
from brendapyrser.parser import EnzymeConditionDict, EnzymePropertyDi

# TODO: Incomplete, need to finish the class
# Also need to know where to get the organism
class BrendaLoader():
    def __init__(self,
                 df: pd.DataFrame = None,
                 ec_column: str = "ec_number",
                 species_column: str = "species",
                 db_file: str = "brenda.sqlite",):
        """
        Initialize the BrendaLoader with the path to the BRENDA database file.
        
        Attributes
        db_file (str): Path to the BRENDA database file.
        df (pd.DataFrame): DataFrame containing the EC numbers and species names.
        ec_column (str): Name of the column containing the EC numbers.
        species (str): Name of the column containing the species names.
        """
        self.data = df[[ec_column, species_column]]
        self.brenda = BRENDA(db_file)


    def process_ec_number(self, ec: str, species: str) -> pd.DataFrame:
        """Process a single EC number and returns a Dataframe with the stability, optimum, and range data"""

        if not self.is_ec_number(ec):
            raise ValueError(f"Invalid EC number: {ec}")
        
        temperature = self.brenda.reactions.get_by_id(ec).temperature
        stability = temperature["stability"]
        
        
        

    def is_ec_number(ec: str) -> bool:
        return re.match(r"^\d+\.\d+\.\d+\.\d+$", ec) is not None
    


    def get_ec_numbers(reaction: Reaction) -> List[str]:
        """Get the EC numbers for a reaction"""
        ec_numbers = []

        for i in range(len(reaction)):
            ec = reaction.__getitem__(i).ec_number
            if is_ec_number(ec):
                ec_numbers.append(ec)

        return ec_numbers

    def get_stability_df(enzyme_dict: EnzymeConditionDict, ec: str) -> pd.DataFrame:
        """Get the stability data for a reaction"""
        tmp = []
        stability_df = pd.DataFrame.from_dict(enzyme_dict['stability'])

        for _, row in stability_df.iterrows():
            for species in row['species']:
                tmp.append([ec, species, row["value"]])

        stability_df = pd.DataFrame(tmp, columns=["ec_number", "species", "stability"])
        return stability_df

    def get_optimum_df(enzyme_dict: EnzymeConditionDict, ec: str) -> pd.DataFrame:
        """Get the optimum data for a reaction"""
        tmp = []
        optimum_df = pd.DataFrame.from_dict(enzyme_dict['optimum'])

        for _, row in optimum_df.iterrows():
            for species in row['species']:
                tmp.append([ec, species, row["value"]])

        optimum_df = pd.DataFrame(tmp, columns=["ec_number", "species", "optimum"])
        return optimum_df

    def get_range_df(enzyme_dict: EnzymeConditionDict, ec: str) -> pd.DataFrame:
        """Get the range data for a reaction"""
        tmp = []
        range_df = pd.DataFrame.from_dict(enzyme_dict['range'])

        for index, row in range_df.iterrows():
            for species in row['species']:
                tmp.append([ ec, species, 
                    row["value"][0] if len(row["value"]) > 1 else None,
                    row["value"][1] if len(row["value"]) > 1 else None])

        range_df = pd.DataFrame(tmp, columns=[
                                    "ec_number", 
                                    "species", 
                                    "range_min", 
                                    "range_max"])
        return range_df

    def get_values_df(enzyme_dict: EnzymePropertyDict, ec: str, type: str) -> pd.DataFrame:
        """Get the values data for a reaction"""
        tmp = []

        for key in enzyme_dict:
            for entry in enzyme_dict[key]:
                tmp.append([ec, entry["species"], entry["value"], type])

        values_df = pd.DataFrame(tmp, columns=["ec_number", "species", "value", "constant_type"])
        return values_df
