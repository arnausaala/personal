import React from 'react';
import { View, Button, StyleSheet, Text } from 'react-native';

export default function Controls({ currentPlayer, onNextTurn, onReset }) {
  return (
    <View style={styles.container}>
      <Text style={styles.turnText}>Turno del jugador {currentPlayer}</Text>
      <View style={styles.buttons}>
        <Button title="Siguiente turno" onPress={onNextTurn} />
        <Button title="Reiniciar partida" color="#888" onPress={onReset} />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    padding: 12,
    alignItems: 'center',
  },
  turnText: {
    fontSize: 16,
    marginBottom: 8,
    fontWeight: 'bold',
  },
  buttons: {
    flexDirection: 'row',
    gap: 10,
  },
});
