/*
 * Copyright (c) 2017 Villu Ruusmann
 *
 * This file is part of JPMML-SparkML
 *
 * JPMML-SparkML is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * JPMML-SparkML is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with JPMML-SparkML.  If not, see <http://www.gnu.org/licenses/>.
 */
package org.jpmml.sparkml.xgboost;

import ml.dmlc.xgboost4j.scala.Booster;
import org.dmg.pmml.DataType;
import org.dmg.pmml.mining.MiningModel;
import org.jpmml.converter.*;
import org.jpmml.sparkml.ModelConverter;
import org.jpmml.xgboost.Learner;
import org.jpmml.xgboost.XGBoostUtil;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

class BoosterUtil {

    private BoosterUtil() {
    }

    static <C extends ModelConverter<?> & HasXGBoostOptions> MiningModel encodeBooster(C converter, Booster booster, Schema schema) {
        byte[] bytes = booster.toByteArray();

        Learner learner;

        try (InputStream is = new ByteArrayInputStream(bytes)) {
            learner = XGBoostUtil.loadLearner(is);
        } catch (IOException ioe) {
            throw new RuntimeException(ioe);
        }


        Schema transformedSchema = transSchema(schema);

        Map<String, Object> options = new LinkedHashMap<>();
        options.put(org.jpmml.xgboost.HasXGBoostOptions.OPTION_COMPACT, converter.getOption(HasXGBoostOptions.OPTION_COMPACT, false));
        options.put(org.jpmml.xgboost.HasXGBoostOptions.OPTION_NTREE_LIMIT, converter.getOption(HasXGBoostOptions.OPTION_NTREE_LIMIT, null));

        return learner.encodeMiningModel(options, transformedSchema);
    }


    static private Schema transSchema(Schema oriSchema) {
        Function<Feature, Feature> function = feature -> {
            if (feature instanceof BinaryFeature) {
                return (BinaryFeature) feature;
            } else {
                return feature.toContinuousFeature(DataType.FLOAT);
            }
        };

        Schema xgbSchema = oriSchema.toTransformedSchema(function);

        //origin
        Label label = xgbSchema.getLabel();
        List<? extends Feature> features = oriSchema.getFeatures();

        //transformed
        Label transformedLabel = new ContinuousLabel(label.getName(), label.getDataType());
        List<? extends Feature> transformedFeatures = features.stream()
                .map(function)
                .collect(Collectors.toList());

        return new Schema(transformedLabel, transformedFeatures);
    }
}