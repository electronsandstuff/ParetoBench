
# "Single Line" Serialization and Deserialization of Problems
Sometimes it is useful to be able to specify problems with all of their parameters in a short single line string. Examples of applications include defining problems for testing optimizers in a config file, referring to problems in log files, and saving problems along with all parameters in file formats. To support this need, ParetoBench includes a simple serialization/deserialization format for problem parameters as well as a standard format for writing the problem's name and list of parameters into a single line string.

## The Serialization Format
The scope of this format is restricted to dicts containing floats, strings, ints, and bools. This is enough to allow the saving / loading of parameters for all of the problem objects without having to define a more complicated standard.

All of the keys and values are written in pairs of the format `<key>=<value>` and are combined in a single string with the pairs separated by commas. White space is allowed around all of the elements (the keys, values, and pairs) and will be removed on deserialization. The keys may only be alphanumeric strings plus the underscore character. For int and float values, they can take on any value which is correctly interpreted by the respective python conversion functions. Floats must contain the decimal point character. Strings must be contained between two double quotes. The double quote character and backslash character must both be escaped with a single backslash character.

## The Problem Line Format
Problems along with their parameters are serialized in the format `NAME (<SERIALIZED PARAMETERS>)`. The name will always be the name of the python class in this library which should also match the literature name for the problem. It must be an alphanumeric string and can include the underscore character. The string "serialized parameters" is generated from the class's parameters using the above described serialization format. White space may be included around and in between any of these objects and will be parsed out.

Not all parameters need to be defined in the serialized parameters. Default values in the classes will be used for any parameters not specified. Problems can also be specified by name only (ie `NAME`) and this corresponds to the problem with all default parameters. The standard also allows for the format `NAME ()` for objects without parameters or all default parameters.

It should be noted that many names may be used to specify the same problem object. This means that the line format should never be used to compare problems or be used as the key to a dictionary for instance as you will end up with duplicates problems.
