import React from 'react';
import { View, StyleSheet } from 'react-native';

export default function Tile({ row, col, isSafe, children }) {
  return (
    <View
      style={[
        styles.tile,
        isSafe ? styles.safe : styles.normal,
        (row + col) % 2 === 0 ? styles.light : styles.dark
      ]}
    >
      {children}
    </View>
  );
}

const styles = StyleSheet.create({
  tile: {
    width: 30,
    height: 30,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 0.5,
    borderColor: '#333',
  },
  safe: {
    backgroundColor: '#aaf',
  },
  normal: {
    backgroundColor: '#eee',
  },
  light: {
    backgroundColor: '#ddd',
  },
  dark: {
    backgroundColor: '#bbb',
  },
});
