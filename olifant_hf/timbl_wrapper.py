"""
Wrapper for TiMBL that uses command-line interface instead of Python bindings.
This is more portable and doesn't require compiled extensions.
"""

import subprocess
import tempfile
import os
from typing import Tuple, Optional


class TimblCLIClassifier:
    """
    TiMBL classifier using command-line interface.

    This wrapper calls the `timbl` command-line tool for classification,
    avoiding the need for compiled Python bindings.
    """

    def __init__(self, fileprefix: str, timbloptions: str, format: str = "Tabbed"):
        self.fileprefix = fileprefix
        self.timbloptions = timbloptions
        self.format = format
        self.ibase_loaded = False

        # Check if timbl is available
        try:
            result = subprocess.run(
                ['timbl', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                print(f"Warning: timbl command returned non-zero: {result.returncode}")
        except FileNotFoundError:
            raise RuntimeError(
                "TiMBL command-line tool not found. Please install it:\n"
                "  macOS: brew install timbl\n"
                "  Ubuntu/Debian: sudo apt install timbl\n"
                "  Alpine: apk add timbl"
            )
        except Exception as e:
            print(f"Warning: Could not verify timbl installation: {e}")

    def load(self):
        """Verify that the .ibase file exists."""
        ibase_path = f"{self.fileprefix}.ibase"
        if not os.path.exists(ibase_path):
            raise FileNotFoundError(f"Instance base not found: {ibase_path}")
        self.ibase_loaded = True
        print(f"TiMBL instance base ready: {ibase_path}")

    def classify(self, features: list, allowtopdistribution: bool = True) -> Tuple[str, str, float]:
        """
        Classify a single instance using TiMBL command-line tool.

        Args:
            features: List of feature values (tokens)
            allowtopdistribution: Whether to get full distribution (not used in CLI mode)

        Returns:
            Tuple of (classlabel, distribution, distance)
        """
        if not self.ibase_loaded:
            raise RuntimeError("Instance base not loaded. Call load() first.")

        # Create temporary test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.test', delete=False) as test_file:
            # Write test instance (features + ?)
            test_line = '\t'.join(str(f) for f in features) + '\t?\n'
            test_file.write(test_line)
            test_path = test_file.name

        # Create temporary output file path
        output_path = test_path + '.out'

        try:
            # Build timbl command
            # Example: timbl -i model.ibase -t test_file.test -o output.out -a4 +D +v+db +v+di
            cmd = [
                'timbl',
                '-i', f'{self.fileprefix}.ibase',
                '-t', test_path,
                '-o', output_path,
                '-F', self.format
            ]

            # Add TiMBL options
            for opt in self.timbloptions.split():
                cmd.append(opt)

            # Add distribution options if not already present
            if '+v+db' not in ' '.join(cmd):
                cmd.extend(['+v+db', '+v+di'])

            # Run TiMBL
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                print(f"TiMBL command failed: {result.stderr}")
                return ("?", "", 1.0)

            # Parse output file
            if os.path.exists(output_path):
                with open(output_path, 'r') as f:
                    lines = f.readlines()

                # Find the classification result line (skip comments)
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Parse output format:
                        # features... correct predicted { dist1 score1, dist2 score2 } distance
                        parts = line.split('\t')

                        if len(parts) >= 2:
                            predicted = parts[-2] if len(parts) > 2 else parts[1]

                            # Extract distribution if present
                            distribution = ""
                            distance = 0.0

                            # Look for distribution in curly braces
                            if '{' in line and '}' in line:
                                dist_start = line.index('{') + 1
                                dist_end = line.index('}')
                                distribution = line[dist_start:dist_end].strip()

                                # Extract distance (last number after })
                                after_dist = line[dist_end+1:].strip()
                                if after_dist:
                                    try:
                                        distance = float(after_dist)
                                    except ValueError:
                                        pass

                            # If no distribution found, use simple format
                            if not distribution:
                                # Try space-separated format
                                space_parts = line.split()
                                if len(space_parts) >= 2:
                                    predicted = space_parts[-2]

                                    # Check for distribution
                                    if '{' in ' '.join(space_parts):
                                        try:
                                            dist_idx = space_parts.index('{')
                                            dist_end_idx = space_parts.index('}')
                                            distribution = ' '.join(space_parts[dist_idx+1:dist_end_idx])
                                            if len(space_parts) > dist_end_idx + 1:
                                                distance = float(space_parts[-1])
                                        except (ValueError, IndexError):
                                            pass

                            return (predicted, distribution, distance)

                # If we get here, parsing failed
                print(f"Failed to parse TiMBL output: {lines}")
                return ("?", "", 1.0)

            else:
                print(f"TiMBL output file not created: {output_path}")
                return ("?", "", 1.0)

        except subprocess.TimeoutExpired:
            print("TiMBL classification timed out")
            return ("?", "", 1.0)
        except Exception as e:
            print(f"Classification error: {e}")
            import traceback
            traceback.print_exc()
            return ("?", "", 1.0)
        finally:
            # Clean up temporary files
            try:
                if os.path.exists(test_path):
                    os.unlink(test_path)
                if os.path.exists(output_path):
                    os.unlink(output_path)
            except Exception as e:
                print(f"Warning: Failed to clean up temp files: {e}")

    def append(self, features: list, classlabel: str):
        """
        Append an instance to memory (not implemented for CLI wrapper).

        For the CLI wrapper, we only support loading pre-trained models.
        To train new models, use the olifant command-line tools.
        """
        print("Warning: append() not supported in CLI mode. Use olifant tools for training.")
