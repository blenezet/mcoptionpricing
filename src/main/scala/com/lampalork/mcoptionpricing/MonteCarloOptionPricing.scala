/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.lampalork.mcoptionpricing

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._ //import the implicit conversions
import org.apache.spark.SparkConf
import org.apache.spark.serializer.{KryoSerializer, KryoRegistrator}
import com.esotericsoftware.kryo.Kryo
import org.apache.commons.math3.random.MersenneTwister
import org.apache.commons.math3.distribution.NormalDistribution
import org.apache.commons.math3.distribution.MultivariateNormalDistribution

import scala.io.Source
import java.io.PrintWriter

case class Instrument(factorWeights: Array[Double], minValue: Double = 0,
  maxValue: Double = Double.MaxValue)

class MyRegistrator extends KryoRegistrator {
  override def registerClasses(kryo: Kryo) {
    kryo.register(classOf[Instrument])
  }
}

case class VanillaOption(spot: Double = 100.0, expiry: Double = 1.0, strike: Double = 100.0, sigma: Double = 0.10, mu: Double = 0.0)

object MonteCarloOptionPricing {
  def main(args: Array[String]) {
    val sparkConf = new SparkConf().setAppName("Monte Carlo Option Pricing")
    sparkConf.set("spark.serializer", classOf[KryoSerializer].getName)
    sparkConf.set("spark.kryo.registrator", classOf[MyRegistrator].getName)
    val sc = new SparkContext(sparkConf)
    
    
    println("*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-")
    println("Monte Carlo Option Pricing")
    println("*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-")
    
    val numSimulations = args(0).toInt
    val parallelism = args(1).toInt    
    
    var option1 = VanillaOption(100,1.0,100, 0.20, 0.05)
    //var instruments = Array(option1)
    
    // Send all instruments to every node
    val broadcastInstrument = sc.broadcast(option1)
    
     // Generate different seeds so that our simulations don't all end up with the same results
    val seed = System.currentTimeMillis()
    val seeds = (seed until seed + parallelism)
    val seedRdd = sc.parallelize(seeds, parallelism)
    
    // Main computation: run simulations and compute aggregate return for each
    val trialsRdd = seedRdd.flatMap(runSimulation(_, numSimulations / parallelism, broadcastInstrument.value))
    
    // Cache the results so that we don't recompute for both of the summarizations below
    trialsRdd.cache()
    
    // Show stock path
    //.foreach(println)
    //trialsRdd.foreach((x: Array[Double]) => println(x.last))
    //val option1payoffs = trialsRdd.map((x: Array[Double]) => x.last)
    val option1payoffs = trialsRdd.map((x: Array[Double]) => math.max(x.last - option1.strike,0))
    //option1payoffs.foreach(println)
    val option1price = math.exp(- option1.mu * option1.expiry) * option1payoffs.mean()
    println("Otion1 Price: " + option1price)
    
    //val varFivePercent = trialsRdd.takeOrdered(math.max(numTrials / 20, 1)).last
    
    // Parse arguments and read input data
    /*val instruments = readInstruments(args(0))
    val numTrials = args(1).toInt
    val parallelism = args(2).toInt
    val factorMeans = readMeans(args(3))
    val factorCovariances = readCovariances(args(4))
    val seed = if (args.length > 5) args(5).toLong else System.currentTimeMillis()

    // Send all instruments to every node
    val broadcastInstruments = sc.broadcast(instruments)

    // Generate different seeds so that our simulations don't all end up with the same results
    val seeds = (seed until seed + parallelism)
    val seedRdd = sc.parallelize(seeds, parallelism)

    // Main computation: run simulations and compute aggregate return for each
    val trialsRdd = seedRdd.flatMap(trialValues(_, numTrials / parallelism,
      broadcastInstruments.value, factorMeans, factorCovariances))

    // Cache the results so that we don't recompute for both of the summarizations below
    trialsRdd.cache()

    // Calculate VaR
    val varFivePercent = trialsRdd.takeOrdered(math.max(numTrials / 20, 1)).last
    println("*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-")
    println("VaR: " + varFivePercent)
    println("*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-")
    */
    // Kernel density estimation
    // val domain = Range.Double(20.0, 60.0, .2).toArray
    // val densities = KernelDensity.estimate(trialsRdd, 0.25, domain)
    // val pw = new PrintWriter("densities.csv")
    // for (point <- domain.zip(densities)) {
    //  pw.println(point._1 + "," + point._2)
    //}
    // pw.close()
  }
  
  def runSimulation(seed: Long, numSimulations: Int, instrument: VanillaOption): Array[Array[Double]] = {
    //val rand = new MersenneTwister(seed)
    //val normal = new NormalDistribution(rand, 0.0, 1.0)
    val numTimeSteps = math.round( instrument.expiry / (1.0/365)).toInt    
    val dt = instrument.expiry / numTimeSteps
    
    val r = new scala.util.Random
    //val phi = Seq.fill(numSimulations * numTimeSteps)(r.nextGaussian)    
    
    val timeSeries = new Array[Array[Double]](numSimulations)
    
    // perform the numSimulations simulations
    for (i <- 0 until numSimulations) {
      
      // generate time serie
      val timeSerie = new Array[Double](numTimeSteps + 1)       
      timeSerie(0) = instrument.spot // init first stock price to spot   
      for (k <- 1 until (numTimeSteps+1)) {
        //val dX = normal.sample()
        val dX = r.nextGaussian()
        //val dX = phi(numSimulations*numTimeSteps + (k-1))
        val dS = instrument.mu * dt + instrument.sigma * math.sqrt(dt) * dX
        timeSerie(k) = timeSerie(k-1) * (1.0 + dS)
      }
      
      // save timeserie
      timeSeries(i) = timeSerie
    }
    
    timeSeries
  }
  
  def trialValues(seed: Long, numTrials: Int, instruments: Seq[Instrument],
      factorMeans: Array[Double], factorCovariances: Array[Array[Double]]): Seq[Double] = {
    val rand = new MersenneTwister(seed)
    val multivariateNormal = new MultivariateNormalDistribution(rand, factorMeans,
      factorCovariances)

    val trialValues = new Array[Double](numTrials)
    for (i <- 0 until numTrials) {
      val trial = multivariateNormal.sample()
      trialValues(i) = trialValue(trial, instruments)
    }
    trialValues
  }

  /**
   * Calculate the full value of the portfolio under particular trial conditions.
   */
  def trialValue(trial: Array[Double], instruments: Seq[Instrument]): Double = {
    var totalValue = 0.0
    for (instrument <- instruments) {
      totalValue += instrumentTrialValue(instrument, trial)
    }
    totalValue
  }

  /**
   * Calculate the value of a particular instrument under particular trial conditions.
   */
  def instrumentTrialValue(instrument: Instrument, trial: Array[Double]): Double = {
    var instrumentTrialValue = 0.0
    var i = 0
    while (i < trial.length) {
      instrumentTrialValue += trial(i) * instrument.factorWeights(i)
      i += 1
    }
    Math.min(Math.max(instrumentTrialValue, instrument.minValue), instrument.maxValue)
  }

  def readInstruments(instrumentsFile: String): Array[Instrument] = {
    val src = Source.fromFile(instrumentsFile)
    // First and second elements are the min and max values for the instrument
    val instruments = src.getLines().map(_.split(",")).map(
      x => new Instrument(x.slice(2, x.length).map(_.toDouble), x(0).toDouble, x(1).toDouble))
    instruments.toArray
  }

  def readMeans(meansFile: String): Array[Double] = {
    val src = Source.fromFile(meansFile)
    val means = src.getLines().map(_.toDouble)
    means.toArray
  }

  def readCovariances(covsFile: String): Array[Array[Double]] = {
    val src = Source.fromFile(covsFile)
    val covs = src.getLines().map(_.split(",")).map(_.map(_.toDouble))
    covs.toArray
  }
}
