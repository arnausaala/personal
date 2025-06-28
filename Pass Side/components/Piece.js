import React from 'react';
import { View, StyleSheet } from 'react-native';

export default function Piece({ player }) {
  const color = player === 1 ? '#ff4d4d' : '#4d94ff';

  return <View style={[styles.piece, { backgroundColor: color }]} />;
}

const styles = StyleSheet.create({
  piece: {
    width: 20,
    height: 20,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#000',
  },
});
