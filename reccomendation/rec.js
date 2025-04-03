// Import required libraries
import Papa from 'papaparse';
import fs from 'fs'; // Note: This is for Node.js. Use appropriate file reading for your environment

/**
 * Load the CSV and parse the ordered song IDs as arrays
 * @param {string} csvFilename - Path to the CSV file
 * @returns {Object} - Object with features as keys and ordered song ID arrays as values
 */
function loadData(csvFilename) {
  // Read the CSV file
  const csvData = fs.readFileSync(csvFilename, 'utf8');
  
  // Parse the CSV data
  const results = Papa.parse(csvData, {
    header: true,
    skipEmptyLines: true
  });
  
  // Create a data object with features as keys and ordered song ID arrays as values
  const data = {};
  results.data.forEach(row => {
    // Parse the string representation of arrays into actual arrays
    data[row.feature] = JSON.parse(row.ordered_song_ids);
  });
  
  return data;
}

/**
 * Find the nearest songs based on feature similarity
 * @param {Object} data - Object with features as keys and ordered song ID arrays as values
 * @param {number} songId - ID of the song to find neighbors for
 * @param {number} numNeighbors - Total number of neighbors to consider
 * @param {Array} groups - Optional array of feature group names to filter by
 * @returns {Array} - Array of [songId, count] pairs representing the most common nearby songs
 */
function findNearestSongs(data, songId, numNeighbors = 1000, groups = null) {
  // Map for counting song occurrences
  const songCounts = new Map();
  
  // Feature group mappings
  const mapping = {
    "basic": [
      "Duration_Seconds", "Tempo_BPM", "Beat_Regularity", "Beat_Density", "Beat_Strength"
    ],
    "pitch": [
      "Estimated_Key", "Key_Confidence", "Average_Pitch", "Pitch_Range", "pYIN_Pitch", "Harmonic_Salience"
    ],
    "spectral": [
      "Average_Spectral_Centroid", "Average_Spectral_Rolloff", "Average_Spectral_Bandwidth",
      "Spectral_Contrast_Mean", "Spectral_Entropy", "Spectral_Flatness", "Tonnetz_Features", "Polynomial_Coefficients"
    ],
    "energy": [
      "RMS_Energy_Mean", "RMS_Energy_Std", "Dynamic_Range", "Crest_Factor", "PCEN_Energy"
    ],
    "harmonic": [
      "Harmonic_Ratio", "Tonal_Energy_Ratio", "Variable_Q_Features", "Reassigned_Features"
    ],
    "rhythm": [
      "Groove_Consistency", "Pulse_Clarity", "Fourier_Tempogram", "Tempogram_Ratio", "Onset_Rate", "Onset_Strength_Mean"
    ],
    "structure": [
      "RQA_Features", "Path_Enhanced_Structure", "HPSS_Separation", "MultipleSegmentation_Boundaries"
    ]
  };
  
  let columnsToUse = [];
  
  // If groups are specified, collect the relevant features
  if (groups) {
    groups.forEach(keyword => {
      const groupFeatures = mapping[keyword.toLowerCase()] || [];
      columnsToUse = [...columnsToUse, ...groupFeatures];
    });
    
    // Process only the features in the selected groups
    for (const [feature, songList] of Object.entries(data)) {
      if (columnsToUse.includes(feature)) {
        processFeature(feature, songList);
      }
    }
  } else {
    // Process all features if no groups are specified
    for (const [feature, songList] of Object.entries(data)) {
      processFeature(feature, songList);
    }
  }
  
  // Helper function to process each feature
  function processFeature(feature, songList) {
    const index = songList.indexOf(songId);
    if (index !== -1) {
      const start = Math.max(0, index - Math.floor(numNeighbors / 2));
      const end = index + Math.floor(numNeighbors / 2);
      const nearestSongs = songList.slice(start, end);
      
      nearestSongs.forEach(id => {
        if (id !== songId) {
          songCounts.set(id, (songCounts.get(id) || 0) + 1);
        }
      });
    }
  }
  
  // Remove the original song ID from the counts
  songCounts.delete(songId);
  
  // Convert the Map to an array of [songId, count] pairs and sort by count (descending)
  const sortedSongs = [...songCounts.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 20); // Get top 20 songs
  
  return sortedSongs;
}

// Example usage
const csvFilename = "path/to/ordered_songs_by_feature.csv"; // Change to your actual path
const songIdToSearch = 0; // Change to the song ID you're searching for

try {
  const data = loadData(csvFilename);
  
  // Without specifying feature groups
  const mostCommonSongs = findNearestSongs(data, songIdToSearch);
  console.log("Most common nearby songs:", mostCommonSongs);
  
  // With specific feature groups
  // const mostCommonSongsWithGroups = findNearestSongs(data, songIdToSearch, 1000, ['basic', 'rhythm']);
  // console.log("Most common nearby songs (basic + rhythm):", mostCommonSongsWithGroups);
} catch (error) {
  console.error("Error:", error.message);
}

// Example for browser environment using fetch
async function loadDataFromWeb(csvUrl) {
  const response = await fetch(csvUrl);
  const csvData = await response.text();
  
  const results = Papa.parse(csvData, {
    header: true,
    skipEmptyLines: true
  });
  
  const data = {};
  results.data.forEach(row => {
    data[row.feature] = JSON.parse(row.ordered_song_ids);
  });
  
  return data;
}

// Usage in browser
// async function init() {
//   const data = await loadDataFromWeb('https://example.com/ordered_songs_by_feature.csv');
//   const mostCommonSongs = findNearestSongs(data, songIdToSearch);
//   console.log("Most common nearby songs:", mostCommonSongs);
// }
// init();