"""
Configuration handling for the GPT model.
"""

import sys
from ast import literal_eval

class Config:
    """
    Configuration class for handling model and training parameters
    """
    def __init__(self, **kwargs):
        # Set all provided keyword arguments as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)
            
    @classmethod
    def from_file(cls, config_file):
        """
        Load configuration from a Python file
        """
        # Create a new dictionary to store configuration
        config_dict = {}
        
        # Read and execute the configuration file
        with open(config_file, 'r') as f:
            config_content = f.read()
            
        # Create a local namespace for execution
        local_vars = {}
        exec(config_content, {}, local_vars)
        
        # Extract variables that don't start with underscore
        for key, value in local_vars.items():
            if not key.startswith('_'):
                config_dict[key] = value
                
        return cls(**config_dict)
    
    def override_from_args(self, args):
        """
        Override configuration from command line arguments
        """
        for arg in args:
            if '=' not in arg:
                # Skip arguments that don't have key=value format
                continue
                
            if not arg.startswith('--'):
                # Skip arguments that don't start with --
                continue
                
            # Parse key and value
            key, val = arg.split('=')
            key = key[2:] # Remove -- prefix
            
            if hasattr(self, key):
                try:
                    # Try to evaluate the value (for numbers, booleans, etc.)
                    attempt = literal_eval(val)
                except (SyntaxError, ValueError):
                    # If evaluation fails, use the string value
                    attempt = val
                    
                # Ensure the types match
                current_val = getattr(self, key)
                if current_val is not None and not isinstance(attempt, type(current_val)):
                    print(f"Warning: Type mismatch for {key}. Expected {type(current_val)}, got {type(attempt)}")
                    
                # Set the attribute
                setattr(self, key, attempt)
                print(f"Overriding: {key} = {attempt}")
            else:
                print(f"Warning: Unknown config key: {key}")
                
        return self
    
    def to_dict(self):
        """
        Convert configuration to dictionary
        """
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    @classmethod
    def from_command_line(cls, default_config=None):
        """
        Create configuration from command line arguments and optional config file
        """
        # Start with default configuration if provided
        config = cls() if default_config is None else default_config
        
        # Process command line arguments
        args = sys.argv[1:]
        config_file = None
        
        # Check if a config file is specified
        for arg in args:
            if '=' not in arg and not arg.startswith('--'):
                config_file = arg
                break
                
        # Load configuration from file if specified
        if config_file is not None:
            print(f"Loading configuration from {config_file}")
            file_config = cls.from_file(config_file)
            
            # Update current configuration with file configuration
            for key, value in file_config.to_dict().items():
                setattr(config, key, value)
                
        # Override with command line arguments
        config.override_from_args(args)
        
        return config